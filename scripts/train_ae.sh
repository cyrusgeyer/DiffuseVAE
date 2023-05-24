# # CelebAMaskHQ training

# Contrastive learning script.
#python main/train_ae.py +dataset=celebamaskhq128/train \
#                     dataset.vae.data.root=''path-to-CelebAHQ-data-root'  \
#                     dataset.vae.data.name='celebamaskhq' \
#                     dataset.vae.training.results_dir=\'vae_celebaHQ_contrastive\' \
#                     dataset.vae.training.chkpt_prefix=\'cmhq128_contrastive\' \
#                     dataset.vae.training.contrastive=True \
#                     dataset.vae.training.c_weight=300 \
#                     dataset.vae.training.max_c_weight=300 \
#                     dataset.vae.training.decay_c_rate=0.0001 \
#                     dataset.vae.training.batch_size=64 \

# Vanilla learning script.
#python main/train_ae.py +dataset=celebamaskhq128/train \
#                     dataset.vae.data.root='path-to-CelebAHQ-data-root' \
#                     dataset.vae.data.name='celebamaskhq' \
#                     dataset.vae.training.results_dir=\'vae_celebaHQ_vanilla\' \
#                     dataset.vae.training.chkpt_prefix=\'cmhq128_vanilla\' \
#                     dataset.vae.training.contrastive=False \
#                     dataset.vae.training.batch_size=64 \