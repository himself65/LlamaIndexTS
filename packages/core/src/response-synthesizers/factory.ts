import { z } from 'zod'
import {
	defaultRefinePrompt,
	defaultTextQAPrompt, defaultTreeSummarizePrompt, type ModuleRecord,
	PromptMixin, type PromptsRecord, type RefinePrompt,
	type TextQAPrompt, type TreeSummarizePrompt
} from '../prompts';
import type { LLM } from '../llms';
import { extractText, streamConverter } from '../utils';
import type { QueryType } from '../query-engine';
import { BaseSynthesizer } from './base-synthesizer'
import { PromptHelper } from '../indices';

const responseModeSchema = z.enum(
	['refine', 'compact', 'simple_summarize', 'generation', 'no_text', 'accumulate', 'compact_accumulate']
)

export type ResponseMode = z.infer<typeof responseModeSchema>

/**
 * A response builder that uses the query to ask the LLM generate a better response using multiple text chunks.
 */
export class Refine extends BaseSynthesizer {
	llm: LLM;
	promptHelper: PromptHelper;
	textQATemplate: TextQAPrompt;
	refineTemplate: RefinePrompt;

	constructor(
		serviceContext?: ServiceContext,
		textQATemplate?: TextQAPrompt,
		refineTemplate?: RefinePrompt,
	) {
		super();

		this.llm = llmFromSettingsOrContext(serviceContext);
		this.promptHelper = promptHelperFromSettingsOrContext(serviceContext);
		this.textQATemplate = textQATemplate ?? defaultTextQAPrompt;
		this.refineTemplate = refineTemplate ?? defaultRefinePrompt;
	}

	protected _getPromptModules(): ModuleRecord {
		return {};
	}

	protected _getPrompts(): {
		textQATemplate: TextQAPrompt;
		refineTemplate: RefinePrompt;
	} {
		return {
			textQATemplate: this.textQATemplate,
			refineTemplate: this.refineTemplate,
		};
	}

	protected _updatePrompts(prompts: {
		textQATemplate: TextQAPrompt;
		refineTemplate: RefinePrompt;
	}): void {
		if (prompts.textQATemplate) {
			this.textQATemplate = prompts.textQATemplate;
		}

		if (prompts.refineTemplate) {
			this.refineTemplate = prompts.refineTemplate;
		}
	}

	getResponse(
		query: ResponseBuilderQuery,
		stream: true,
	): Promise<AsyncIterable<string>>;
	getResponse(query: ResponseBuilderQuery, stream?: false): Promise<string>;
	async getResponse(
		{ query, textChunks, prevResponse }: ResponseBuilderQuery,
		stream?: boolean,
	): Promise<AsyncIterable<string> | string> {
		let response: AsyncIterable<string> | string | undefined = prevResponse;

		for (let i = 0; i < textChunks.length; i++) {
			const chunk = textChunks[i];
			const lastChunk = i === textChunks.length - 1;
			if (!response) {
				response = await this.giveResponseSingle(
					query,
					chunk,
					!!stream && lastChunk,
				);
			} else {
				response = await this.refineResponseSingle(
					response as string,
					query,
					chunk,
					!!stream && lastChunk,
				);
			}
		}

		return response ?? "Empty Response";
	}

	private async giveResponseSingle(
		query: QueryType,
		textChunk: string,
		stream: boolean,
	): Promise<AsyncIterable<string> | string> {
		const textQATemplate: TextQAPrompt = this.textQATemplate.partialFormat({
			query: extractText(query),
		});
		const textChunks = this.promptHelper.repack(textQATemplate, [textChunk]);

		let response: AsyncIterable<string> | string | undefined = undefined;

		for (let i = 0; i < textChunks.length; i++) {
			const chunk = textChunks[i];
			const lastChunk = i === textChunks.length - 1;
			if (!response) {
				response = await this.complete({
					prompt: textQATemplate.format({
						context: chunk,
					}),
					stream: stream && lastChunk,
				});
			} else {
				response = await this.refineResponseSingle(
					response as string,
					query,
					chunk,
					stream && lastChunk,
				);
			}
		}

		return response as AsyncIterable<string> | string;
	}

	// eslint-disable-next-line max-params
	private async refineResponseSingle(
		initialReponse: string,
		query: QueryType,
		textChunk: string,
		stream: boolean,
	) {
		const refineTemplate: RefinePrompt = this.refineTemplate.partialFormat({
			query: extractText(query),
		});

		const textChunks = this.promptHelper.repack(refineTemplate, [textChunk]);

		let response: AsyncIterable<string> | string = initialReponse;

		for (let i = 0; i < textChunks.length; i++) {
			const chunk = textChunks[i];
			const lastChunk = i === textChunks.length - 1;
			response = await this.complete({
				prompt: refineTemplate.format({
					context: chunk,
					existingAnswer: response as string,
				}),
				stream: stream && lastChunk,
			});
		}
		return response;
	}

	async complete(params: {
		prompt: string;
		stream: boolean;
	}): Promise<AsyncIterable<string> | string> {
		if (params.stream) {
			const response = await this.llm.complete({ ...params, stream: true });
			return streamConverter(response, (chunk) => chunk.text);
		} else {
			const response = await this.llm.complete({ ...params, stream: false });
			return response.text;
		}
	}
}

/**
 * CompactAndRefine is a slight variation of Refine that first compacts the text chunks into the smallest possible number of chunks.
 */
export class CompactAndRefine extends Refine {
	getResponse(
		query: ResponseBuilderQuery,
		stream: true,
	): Promise<AsyncIterable<string>>;
	getResponse(query: ResponseBuilderQuery, stream?: false): Promise<string>;
	async getResponse(
		{ query, textChunks, prevResponse }: ResponseBuilderQuery,
		stream?: boolean,
	): Promise<AsyncIterable<string> | string> {
		const textQATemplate: TextQAPrompt = this.textQATemplate.partialFormat({
			query: extractText(query),
		});
		const refineTemplate: RefinePrompt = this.refineTemplate.partialFormat({
			query: extractText(query),
		});

		const maxPrompt = getBiggestPrompt([textQATemplate, refineTemplate]);
		const newTexts = this.promptHelper.repack(maxPrompt, textChunks);
		const params = {
			query,
			textChunks: newTexts,
			prevResponse,
		};
		if (stream) {
			return super.getResponse(
				{
					...params,
				},
				true,
			);
		}
		return super.getResponse(params);
	}
}

/**
 * TreeSummarize repacks the text chunks into the smallest possible number of chunks and then summarizes them, then recursively does so until there's one chunk left.
 */
export class TreeSummarize extends PromptMixin implements ResponseBuilder {
	llm: LLM;
	promptHelper: PromptHelper;
	summaryTemplate: TreeSummarizePrompt;

	constructor(
		serviceContext?: ServiceContext,
		summaryTemplate?: TreeSummarizePrompt,
	) {
		super();

		this.llm = llmFromSettingsOrContext(serviceContext);
		this.promptHelper = promptHelperFromSettingsOrContext(serviceContext);
		this.summaryTemplate = summaryTemplate ?? defaultTreeSummarizePrompt;
	}

	protected _getPromptModules(): ModuleRecord {
		return {};
	}

	protected _getPrompts(): { summaryTemplate: TreeSummarizePrompt } {
		return {
			summaryTemplate: this.summaryTemplate,
		};
	}

	protected _updatePrompts(prompts: {
		summaryTemplate: TreeSummarizePrompt;
	}): void {
		if (prompts.summaryTemplate) {
			this.summaryTemplate = prompts.summaryTemplate;
		}
	}

	getResponse(
		query: ResponseBuilderQuery,
		stream: true,
	): Promise<AsyncIterable<string>>;
	getResponse(query: ResponseBuilderQuery, stream?: false): Promise<string>;
	async getResponse(
		{ query, textChunks }: ResponseBuilderQuery,
		stream?: boolean,
	): Promise<AsyncIterable<string> | string> {
		if (!textChunks || textChunks.length === 0) {
			throw new Error("Must have at least one text chunk");
		}

		// Should we send the query here too?
		const packedTextChunks = this.promptHelper.repack(
			this.summaryTemplate,
			textChunks,
		);

		if (packedTextChunks.length === 1) {
			const params = {
				prompt: this.summaryTemplate.format({
					context: packedTextChunks[0],
					query: extractText(query),
				}),
			};
			if (stream) {
				const response = await this.llm.complete({ ...params, stream });
				return streamConverter(response, (chunk) => chunk.text);
			}
			return (await this.llm.complete(params)).text;
		} else {
			const summaries = await Promise.all(
				packedTextChunks.map((chunk) =>
					this.llm.complete({
						prompt: this.summaryTemplate.format({
							context: chunk,
							query: extractText(query),
						}),
					}),
				),
			);

			const params = {
				query,
				textChunks: summaries.map((s) => s.text),
			};
			if (stream) {
				return this.getResponse(
					{
						...params,
					},
					true,
				);
			}
			return this.getResponse(params);
		}
	}
}

export function getResponseSynthesizer(
	mode: ResponseMode
) {

}