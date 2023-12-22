using Pkg.Artifacts

function __init__()
    artifact_toml = joinpath(@__DIR__, "../Artifacts.toml")

    MLIRdist_hash = artifact_hash("MLIRdist", artifact_toml)

    if MLIRdist_hash == nothing || !artifact_exists(MLIRdist_hash)
        MLIRdist_hash = create_artifact() do artifact_dir
            @info "Downloading MLIR distribution (this can take a while ~700MB)"
            temp = download("https://github.com/makslevental/mlir-wheels/releases/download/latest/mlir-18.0.0.2023121501+bf2b035e-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
            @info "Unzipping $temp to $artifact_dir"
            run(`unzip -q $temp -d $artifact_dir`)
        end
        if MLIRdist_hash == nothing # should not trigger for users.
            bind_artifact!(
                artifact_toml,
                "MLIRdist",
                MLIRdist_hash)
        end
    end
end
