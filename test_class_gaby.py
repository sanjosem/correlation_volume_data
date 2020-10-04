import gaby_correlation as gcorr
# data_file = '/bb/s/sanjosem/sanjosem/vol_DNS_8deg/volume_416_74_257.part_000.hdf5'
data_file = './volume_416_74_257.part_001.hdf5'
mesh_file = './interpolation_3d_grid_dims.hdf5'
a = gcorr.gaby_correlation(data_file,mesh_file,136)
a.compute_disp()
layer = 23 
R11,R22=a.compute_R11_R22(layer)
r1,r2,r3 = a.get_ref_layer_radius(layer)
