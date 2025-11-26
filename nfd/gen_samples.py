

'''
Generate 3D samples of a given class using a pretrained model.
Will output triplanes as .npy files and meshes as .ply files. 
'''


import neural_field_diffusion.scripts.image_sample as image_sample
import triplane_decoder.visualize as visualize
import mesh_to_pointcloud as m2pc
import argparse
from argparse import Namespace
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser(description='Generate a set of triplanes and their corresponding meshes')
    parser.add_argument('--resolution', type=str, default=128, required=False,
                    help='Triplane resolution.')
    parser.add_argument('--ddpm_ckpt', type=str,
                    help='DDPM checkpoint.', required=True)
    parser.add_argument('--decoder_ckpt', type=str,
                    help='Decoder checkpoint.', required=True)
    parser.add_argument('--stats_dir', type=str,
                    help='Normalization statistics to use.', required=True)
    parser.add_argument('--save_dir', type=str, default='./samples',
                    help='Where to save generated samples.', required=False)
    parser.add_argument('--num_samples', type=int, default=8,
                    help='How many samples to generate.', required=False)
    parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for the DDPM. Use a smaller batch size if GPU memory is low.', required=False)
    parser.add_argument('--num_steps', type=int, default=1000,
                    help='Number of steps to take in deniosing process.', required=False)
    parser.add_argument('--shape_resolution', type=int, default=128,
                    help='Resolution at which to decode shapes.', required=False)
    parser.add_argument('--generate_pointclouds', action='store_true',
                    help='Generate point cloud dataset from meshes')
    parser.add_argument('--pc_num_points', type=int, default=2048,
                    help='Number of points per cloud (default: 2048, standard for PointNet)')
    parser.add_argument('--pc_variants', type=int, default=5,
                    help='Number of variants per mesh (with different noise/occlusion)')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Generate triplane samples using DDPM with default arguments
    ddpm_args = Namespace(
        clip_denoised=True, num_samples=args.num_samples, batch_size=args.batch_size, use_ddim=False, model_path=args.ddpm_ckpt, stats_dir=args.stats_dir, 
        explicit_normalization=True, save_dir=args.save_dir, save_intermediate=False, save_timestep_interval=20, image_size=args.resolution, num_channels=256, 
        num_res_blocks=2, num_heads=4, num_heads_upsample=-1, num_head_channels=64, attention_resolutions='32,16,8', channel_mult='', dropout=0.1, class_cond=False, 
        use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=True, use_fp16=True, use_new_attention_order=False, in_out_channels=96, learn_sigma=True, 
        diffusion_steps=1000, noise_schedule='linear', timestep_respacing=str(args.num_steps), use_kl=False, predict_xstart=False, rescale_timesteps=False, 
        rescale_learned_sigmas=False
    )
    samples = image_sample.main(args=ddpm_args)  # Run ddpm

    # Convert samples to a directory of .npy triplanes
    os.makedirs(f'{args.save_dir}/triplanes', exist_ok=True)

    samples = np.transpose(samples, [0, 3, 1, 2])
    
    os.makedirs(f'{args.save_dir}/triplane_images', exist_ok=True)

    for idx, triplane in enumerate(samples):
        save_path = f'{args.save_dir}/triplanes/{idx}.npy'
        print(f'saving to {save_path}...')
        np.save(save_path, triplane)

        # Save triplane images
        reshaped_triplane = triplane.reshape(3, 32, 128, 128)
        for plane_idx in range(3):
            # Take first 3 channels for RGB visualization
            plane_img = reshaped_triplane[plane_idx, :3, :, :] # (3, 128, 128)
            plane_img = np.transpose(plane_img, (1, 2, 0)) # (128, 128, 3)
            
            # Normalize to [0, 1]
            plane_img = (plane_img - plane_img.min()) / (plane_img.max() - plane_img.min() + 1e-8)
            
            image_save_path = f'{args.save_dir}/triplane_images/{idx}_plane_{plane_idx}.png'
            plt.imsave(image_save_path, plane_img)

    os.system(f'rm {args.save_dir}/*.npz')
    
    # Decode triplane samples
    os.makedirs(f'{args.save_dir}/objects', exist_ok=True)
    
    for idx, triplane in enumerate(samples):
        print(f'Decoding triplane {idx}...')
        decoder_args = Namespace(
            input=f'{args.save_dir}/triplanes/{idx}.npy', output=f'{args.save_dir}/objects/{idx}.obj',
            model_path=args.decoder_ckpt, res=args.shape_resolution
        )
        visualize.main(args=decoder_args)  # Run decoder

    # Generate point clouds from meshes (optional)
    if args.generate_pointclouds:
        print('\nConverting meshes to point clouds...')
        os.makedirs(f'{args.save_dir}/pointclouds', exist_ok=True)
        
        successful = 0
        failed = 0
        
        for idx in range(len(samples)):
            mesh_path = f'{args.save_dir}/objects/{idx}.obj'
            pc_output_dir = f'{args.save_dir}/pointclouds/object_{idx}'
            
            result = m2pc.process_mesh_to_dataset(
                mesh_path=mesh_path,
                output_dir=pc_output_dir,
                num_points=args.pc_num_points,
                num_variants=args.pc_variants,
                add_noise=True,
                generate_partial=True
            )
            
            if result:
                successful += 1
            else:
                failed += 1
        
        print(f'\nPoint cloud generation complete!')
        print(f'  Successfully processed: {successful}/{len(samples)} meshes')
        if failed > 0:
            print(f'  Failed: {failed}/{len(samples)} meshes')

    print('\nDone!')
    


if __name__ == "__main__":
    main()
