# tanrenai-gpu

GPU inference server: thin Go HTTP API wrapping llama.cpp's `llama-server`
with CUDA support. Publishes as `harbor.floret.dev/tanrenai/tanrenai-gpu:latest`
on the private Harbor registry; runs on Vast.ai instances provisioned by
`tanrenai-platform`, and embedded by the `tanrenai` CLI in `--local` mode.

## Build

```sh
go build ./...
go test ./...
```

## Docker

```sh
docker login harbor.floret.dev
docker buildx build --platform linux/amd64 \
  -t harbor.floret.dev/tanrenai/tanrenai-gpu:latest \
  -f docker/Dockerfile --push .
```

`CUDA_ARCHITECTURES` build-arg controls which GPU SMs get compiled kernels.
Default covers Ampere datacenter (A100), Ampere consumer (RTX 30xx / A10 /
A40 / A6000), Ada (RTX 4090 / L40 / L40S), Hopper (H100), and Blackwell.

## Consumers

- `tanrenai-platform` — provisions instances running this image.
- `tanrenai` CLI — imports `pkg/serve` for `--local` mode.
- `tanrenai-infra` — unchanged; references the Docker Hub tag directly.
