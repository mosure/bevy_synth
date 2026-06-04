# burn_flux

Burn model components for Flux-family image-latent encoders used by the synthesis
pipelines in this workspace.

The initial implementation targets the TripoSplat Flux2 VAE encoder path:
`Flux2VaeEncoder` maps a prepared RGB image tensor in `[-1, 1]` to the
`[batch, tokens, 128]` latent-conditioning sequence used by the TripoSplat flow
model.
