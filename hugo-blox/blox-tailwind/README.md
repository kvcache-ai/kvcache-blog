# Hugo Blox Tailwind UI

The Hugo Blox web framework utilising [TailwindCSS](https://tailwindcss.com/) and its [typography plugin](https://tailwindcss.com/docs/typography-plugin) for styling components.

See the Blog Starter Template for an example using this module.

## Development

```bash
hugo mod npm pack
pnpm install
```

This module pins dependency versions via `pnpm.overrides` in `package.json` (see `pnpm-lock.yaml`), so use `pnpm` rather than `npm`/`yarn` to install dependencies here.

You need to add this to your `config.yaml` to enable Hugo's Tailwind integration:

```yaml
build:
  writeStats: true
```
