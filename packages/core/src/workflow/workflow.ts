import { Settings } from '../global';
import { Context, type ContextParams, type StepFunction } from './context';
import { StartEvent, StopEvent, WorkflowEvent } from './events';

export type WorkflowParams = {
	verbose?: boolean;
	timeout?: number;
	validate?: boolean;
};

export class WorkflowTemplate<
	Start = string,
	Stop = string,
	Ctx extends new (params: ContextParams<Start>) => any = typeof Context<Start>,
> {
	#steps: Map<
		StepFunction<Start, Stop, Ctx, any>,
		{
			inputs: (typeof WorkflowEvent<any>)[];
			outputs: (typeof WorkflowEvent<any>)[] | undefined
		}
	> = new Map();
	#verbose: boolean = false;
	#timeout: number | null = null;

	constructor (
		params: WorkflowParams = {}
	) {
		this.#verbose = params.verbose ?? Settings.debug ?? false;
		this.#timeout = params.timeout ?? null;
	}

	static generate<Start, Stop, Ctx extends new (...args: any[]) => any> (): new (
		params: WorkflowParams
	) => WorkflowTemplate<Start, Stop, Ctx>;
	static generate<Ctx extends new (...args: any[]) => any> (
		context: Ctx
	): new (params: WorkflowParams) => WorkflowTemplate<string, string, Ctx>
	static generate<Start, Stop, Ctx extends new (...args: any[]) => any> (
		context: Ctx
	): new (params: WorkflowParams) => WorkflowTemplate<Start, Stop, Ctx>
	static generate<Start, Stop, Ctx extends new (...args: any[]) => any> (
		startEvent: typeof StartEvent<Start>,
		stopEvent: typeof StopEvent<Stop>,
		context: Ctx
	): new (params: WorkflowParams) => WorkflowTemplate<Start, Stop, Ctx>
	static generate<Start, Stop, Ctx extends new (...args: any[]) => any> (
		a?: any,
		b?: any,
		c?: any
	): new (params: WorkflowParams) => WorkflowTemplate<Start, Stop, Ctx> {
		const ctx = arguments.length === 1 ? a : c;
		if (!(ctx.prototype instanceof Context)) {
			throw new Error('Context must be a subclass of Context');
		}

		class Workflow extends WorkflowTemplate {}

		const proto = Workflow.prototype as any;
		proto.run = function(...args: any[]) {
			return WorkflowTemplate.prototype.run.call(this, args[0], ctx);
		};
		return Workflow as any;
	}

	addStep<D extends Record<string, any>> (
		eventType: typeof WorkflowEvent<D> | (typeof WorkflowEvent<D>)[],
		method: StepFunction<Start, Stop, Ctx, WorkflowEvent<D>>,
		params: {
			outputs?: typeof WorkflowEvent<any> | (typeof WorkflowEvent<any>)[]
		} = {}
	) {
		const inputs = Array.isArray(eventType) ? eventType : [eventType];
		const outputs = params.outputs
			? Array.isArray(params.outputs)
				? params.outputs
				: [params.outputs]
			: undefined;
		this.#steps.set(method, { inputs, outputs });
	}

	hasStep (step: StepFunction<Start, Stop, Ctx, any>): boolean {
		return this.#steps.has(step);
	}

	run (event: StartEvent<Start> | Start): Ctx extends Context
		? Context extends Ctx ? never : Context<Start, Stop>
		: Context<Start, Stop>;
	run<
		Ctx extends new (params: ContextParams<Start>) => any
	> (
		event: StartEvent<Start> | Start,
		Context: Ctx): Ctx extends InstanceType<infer R> ? R : never;
	run<
		CtxInstance extends Context<Start>,
		Ctx extends new (params: ContextParams<Start>) => CtxInstance,
	> (event: StartEvent<Start> | Start, CurrentContext?: Ctx): CtxInstance {
		const startEvent: StartEvent<Start> =
			event instanceof StartEvent ? event : new StartEvent({ input: event });

		const C: any = CurrentContext ?? Context;

		return new C({
			startEvent,
			steps: new Map(this.#steps),
			timeout: this.#timeout,
			verbose: this.#verbose
		});
	}
}
