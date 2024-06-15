module github.com/HugoBlox/theme-documentation

go 1.15

require (
	github.com/HugoBlox/hugo-blox-builder/modules/blox-plugin-netlify v1.1.2-0.20240217212918-ae7f0c597978
	github.com/HugoBlox/hugo-blox-builder/modules/blox-tailwind v0.2.1-0.20240602133901-492e343c2a33
)

replace github.com/HugoBlox/hugo-blox-builder/modules/blox-tailwind => ./hugo-blox/blox-tailwind/
replace github.com/HugoBlox/hugo-blox-builder/modules/blox-plugin-netlify => ./hugo-blox/blox-plugin-netlify/