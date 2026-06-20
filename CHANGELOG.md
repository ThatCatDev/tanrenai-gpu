## [1.9.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.8.0...v1.9.0) (2026-06-20)

### Features

* **runner:** capture llama-server crash reason and surface it to requests ([#16](https://github.com/ThatCatDev/tanrenai-gpu/issues/16)) ([ce66a38](https://github.com/ThatCatDev/tanrenai-gpu/commit/ce66a389c215b39a0bc3000c96e9add250e96486))

## [1.8.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.7.1...v1.8.0) (2026-06-20)

### Features

* **chat:** surface mid-stream aborts instead of closing silently ([#15](https://github.com/ThatCatDev/tanrenai-gpu/issues/15)) ([4617a1f](https://github.com/ThatCatDev/tanrenai-gpu/commit/4617a1f78c49820560d3a1b6daf160e7b002fb2a))

## [1.7.1](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.7.0...v1.7.1) (2026-06-03)

### Bug Fixes

* **server:** /api/load reports per-user context, not total ([#14](https://github.com/ThatCatDev/tanrenai-gpu/issues/14)) ([cce09f5](https://github.com/ThatCatDev/tanrenai-gpu/commit/cce09f5599f719f55623642fc3ff322e9b23fdd9))

## [1.7.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.6.0...v1.7.0) (2026-06-03)

### Features

* **server:** expose loaded model + slot capacity at /v1/status ([#13](https://github.com/ThatCatDev/tanrenai-gpu/issues/13)) ([de181ac](https://github.com/ThatCatDev/tanrenai-gpu/commit/de181ac5a209f59e51ef2c9f3259e1b27456c1b7))

## [1.6.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.5.0...v1.6.0) (2026-06-03)

### Features

* **server:** multi-user context slots (--ctx-per-user) ([#12](https://github.com/ThatCatDev/tanrenai-gpu/issues/12)) ([299da6e](https://github.com/ThatCatDev/tanrenai-gpu/commit/299da6ea05fa2658ac17c06e3f0e0867965ad00d))

## [1.5.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.4.1...v1.5.0) (2026-06-01)

### Features

* **server:** VRAM-aware context sizing + context shift by default ([#11](https://github.com/ThatCatDev/tanrenai-gpu/issues/11)) ([e918c2c](https://github.com/ThatCatDev/tanrenai-gpu/commit/e918c2cb375ce3e6ab142e1e6c2a01448df95687)), closes [#9](https://github.com/ThatCatDev/tanrenai-gpu/issues/9)

## [1.4.1](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.4.0...v1.4.1) (2026-05-24)

### Bug Fixes

* **server:** cap auto-detected context size at 32K ([#9](https://github.com/ThatCatDev/tanrenai-gpu/issues/9)) ([dd24e4b](https://github.com/ThatCatDev/tanrenai-gpu/commit/dd24e4bfe7f72db4ce35a9eca439ebc2aab5f99f))

## [1.4.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.3.0...v1.4.0) (2026-05-21)

### Features

* **pull:** auto-derive on-disk basename from hf:// URI ([#8](https://github.com/ThatCatDev/tanrenai-gpu/issues/8)) ([dbeaf75](https://github.com/ThatCatDev/tanrenai-gpu/commit/dbeaf75709fb408c179dd0ddbf22b586489a2f2e))

## [1.3.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.2.0...v1.3.0) (2026-05-14)

### Features

* dynamic parallel concurrency + buildinfo-stamped version ([#7](https://github.com/ThatCatDev/tanrenai-gpu/issues/7)) ([12a0464](https://github.com/ThatCatDev/tanrenai-gpu/commit/12a0464a3b1a03657f8eb3b66a25537d10b5c600))

## [1.2.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.1.2...v1.2.0) (2026-05-14)

### Features

* **download:** parallel range GETs for large GGUF pulls ([#6](https://github.com/ThatCatDev/tanrenai-gpu/issues/6)) ([f2702b5](https://github.com/ThatCatDev/tanrenai-gpu/commit/f2702b58b47f2f4e420bcbfb5a3bc549598207a1))

## [1.1.2](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.1.1...v1.1.2) (2026-05-10)

### Bug Fixes

* **docker:** build llama-server with portable CPU baseline ([#5](https://github.com/ThatCatDev/tanrenai-gpu/issues/5)) ([0345a23](https://github.com/ThatCatDev/tanrenai-gpu/commit/0345a23a787c6d920d0404501a279f5e5b5f7636))

## [1.1.1](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.1.0...v1.1.1) (2026-05-10)

### Bug Fixes

* **runner:** pass --fit off explicitly when FitVRAM is disabled ([#4](https://github.com/ThatCatDev/tanrenai-gpu/issues/4)) ([bd65fce](https://github.com/ThatCatDev/tanrenai-gpu/commit/bd65fce609b4854e9564ee1bd4a6e227534c9762))

## [1.1.0](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.0.2...v1.1.0) (2026-05-10)

### Features

* **pull:** accept name field on /api/pull to override saved filename ([#3](https://github.com/ThatCatDev/tanrenai-gpu/issues/3)) ([de6dd37](https://github.com/ThatCatDev/tanrenai-gpu/commit/de6dd3705dfb56016aa8fe58fcd9ca31d5f27dc6))

## [1.0.2](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.0.1...v1.0.2) (2026-04-20)

### Bug Fixes

* **server:** drop auto-cpu-moe; full offload is the right default ([#2](https://github.com/ThatCatDev/tanrenai-gpu/issues/2)) ([878e1c7](https://github.com/ThatCatDev/tanrenai-gpu/commit/878e1c7cfe6c1b36b303fff2178e17e90026b9c3))

## [1.0.1](https://github.com/ThatCatDev/tanrenai-gpu/compare/v1.0.0...v1.0.1) (2026-04-20)

### Bug Fixes

* **server:** drop auto-FitVRAM for MoE models ([cc5d736](https://github.com/ThatCatDev/tanrenai-gpu/commit/cc5d73610979009e04f81dbdf8deca37305266b4))

## 1.0.0 (2026-04-19)

### Features

* initial extraction from tanrenai monorepo ([7b0c039](https://github.com/ThatCatDev/tanrenai-gpu/commit/7b0c039d525dc7386dee55ef8fdaa87bac93d19a))
