using PackageCompiler

create_sysimage([:MDToolbox, :Distances, :ArgParse], sysimage_path="gauss_image.so", precompile_execution_file="precompile_plots.jl")

# run code with image:
# julia -J gauss_image.so --trace-compile=stderr gauss_linking.jl -f XXX -p YYY
# --trace-compile to see which packing are compiling prior
