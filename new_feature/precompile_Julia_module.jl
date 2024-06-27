using PackageCompiler

# create_sysimage(
#     [:MDToolbox, :Distances, :ArgParse],
#     sysimage_path="gauss_image.so",
#     precompile_execution_file="gauss_linking.jl"
# )
create_sysimage([:BioStructures, :Distances, :ArgParse], sysimage_path="gauss_image.so")
"""
 run code with image:
 julia -J gauss_image.so --trace-compile=stderr gauss_linking.jl -f XXX -p YYY
 --trace-compile to see which packing are compiling prior


 In gauss_linking.jl: we do not specify filetype of filename that load to mdtoolbox. hence package compiler raise error 
 """