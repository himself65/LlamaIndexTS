import { swc, defineRollupSwcOption } from 'rollup-plugin-swc3'
import commonjs from '@rollup/plugin-commonjs'
import { nodeResolve } from '@rollup/plugin-node-resolve'
import { JscTarget } from '@swc/core'
import { RollupOptions } from 'rollup'
import { dependencies } from './package.json'

import { readdirSync } from 'node:fs'

const noBundleExternal = Object.keys(dependencies)
const fileOrDirs = readdirSync('./src')

const skipDir = ['callbacks', 'outputParsers', 'memory', 'engines', 'tests']
const dtsOutput = new Map<string, string>()
const input = fileOrDirs.reduce((input, fileOrDir) => {
  if (fileOrDir.endsWith('.ts')) {
    dtsOutput.set(`./src/${fileOrDir}`, './dist/' + fileOrDir.replace('.ts', '.d.ts'))
    return {
      ...input,
      [fileOrDir.replace('.ts', '')]: `./src/${fileOrDir}`
    }
  } else {
    if (skipDir.includes(fileOrDir)) {
      return input
    }
    dtsOutput.set(`./src/${fileOrDir}/index.ts`, './dist/' + fileOrDir + '/index.d.ts')
    return {
      ...input,
      [fileOrDir]: `./src/${fileOrDir}/index.ts`
    }
  }
  return input
}, {} as Record<string, string>)

const outputMatrix = (config: {
  format: 'es' | 'cjs',
  minify: boolean,
  target: JscTarget,
}): RollupOptions[] => {
  return [
    {
      input,
      output: config.format === 'es' ? {
        dir: 'dist/esm',
        format: 'es'
      } : {
        dir: 'dist',
        format: 'cjs'
      },
      plugins: [
        commonjs({
          esmExternals: true
        }),
        nodeResolve({
          exportConditions: ['import', 'module', 'require', 'default']
        }),
        swc(defineRollupSwcOption({
          jsc: {
            parser: {
              syntax: 'typescript',
              tsx: false
            },
            target: config.target,
            minify: config.minify
              ? { compress: { unsafe: true }, mangle: true, module: true }
              : undefined
          },
          minify: config.minify,
          module: {
            type: 'es6'
          }
        })),
      ],
      external: (id) => {
        return (
          noBundleExternal.includes(id)
          || noBundleExternal.some(dep => id.startsWith(`${dep}/`))
        );
      }
    }
  ]
}

const buildConfig: RollupOptions[] = [
  outputMatrix({
    format: 'es',
    target: 'es2022',
    minify: false
  }),
  outputMatrix({
    format: 'cjs',
    target: 'es2018',
    minify: false
  }),
].flat()

export default buildConfig
