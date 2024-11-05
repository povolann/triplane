## Pretrained models paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = '/home/anya/Programs/triplane/checkpoints'
#embedding_base_dir = '/home/anya/Programs/triplane/projector/PTI/embeddings'
#experiments_output_dir = '/home/anya/Programs/EG3D-projector/eg3d/projector/PTI/output' # maybe not needed

# Dirs for input files
input_data_path = '/home/anya/Programs/triplane/projector_test_data/'
#base_dir = '/home/anya/Programs/triplane/' # maybe not needed, I have outdir_proj

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

# Dirs for my models
eg3d_ffhq_pkl = '/home/anya/Programs/triplane/networks/name.pkl' # this needs to be changed for "triplane"