total_bones = 24

# motion weights volume
mweight_volume_embedding_size = 256
mweight_volume_volume_size = 32
mweight_volume_dst_voxel_size = 0.0625

# canonical mlp
canonical_mlp_mlp_depth = 8
canonical_mlp_mlp_width = 256
canonical_mlp_multires = 10
canonical_mlp_i_embed = 0

# non-rigid motion mlp
non_rigid_motion_mlp_condition_code_size = 69
non_rigid_motion_mlp_mlp_width = 128
non_rigid_motion_mlp_mlp_depth = 6
non_rigid_motion_mlp_skips = [4]
non_rigid_motion_mlp_multires = 6
non_rigid_motion_mlp_i_embed = 0

# pose decoder
pose_decoder_embedding_size = 69
pose_decoder_mlp_width = 256
pose_decoder_mlp_depth = 4