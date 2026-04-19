# tanrenai-gpu

GPU inference server: thin Go HTTP API wrapping llama.cpp's `llama-server`
with CUDA support. Publishes as `thatcatdev/tanrenai-gpu:latest` on Docker
Hub; runs on Vast.ai instances provisioned by `tanrenai-platform`, and
embedded by the `tanrenai` CLI in `--local` mode.

## Build

```sh
go build ./...
go test ./...
```

## Docker

```sh
docker buildx build --platform linux/amd64 \
  -t thatcatdev/tanrenai-gpu:latest \
  -f docker/Dockerfile --push .
```

`CUDA_ARCHITECTURES` build-arg controls which GPU SMs get compiled kernels.
Default covers Ampere datacenter (A100), Ampere consumer (RTX 30xx / A10 /
A40 / A6000), Ada (RTX 4090 / L40 / L40S), Hopper (H100), and Blackwell.

## Consumers

- `tanrenai-platform` — provisions instances running this image.
- `tanrenai` CLI — imports `pkg/serve` for `--local` mode.
- `tanrenai-infra` — unchanged; references the Docker Hub tag directly.
