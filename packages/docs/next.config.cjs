const withNextra = require('nextra')({
	latex: true,
	search: {
		codeblocks: false
	},
	theme: 'nextra-theme-docs',
	themeConfig: './theme.config.jsx',
	mdxBaseDir: './mdx',
	mdxOptions: {
		providerImportSource: 'nextra-theme-docs'
	}
})

export default withNextra({
	reactStrictMode: true
})