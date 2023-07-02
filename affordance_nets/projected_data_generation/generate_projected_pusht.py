import matplotlib.pyplot as plt
import torch
from affordance_nets.common.replay_buffer import ReplayBuffer



if __name__ == '__main__':
    import os
    from affordance_nets.utils.directory_utils import get_data_dir

    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')

    replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['img', 'state', 'action'])


    print(replay_buffer)

    k = 0
    for k in range(replay_buffer.data['img'].shape[0]):
        img = replay_buffer.data['img'][k,...]
        agent_pos = replay_buffer.data['state'][k,:2]*(96/512)
        agent_act = replay_buffer.data['action'][k,:2]*(96/512)


        img2 = img.copy()
        img2[int(agent_pos[1]), int(agent_pos[0])] = [0, 0, 0]

        H = 16
        for l in range(1, H):
            agent_pos_pred = replay_buffer.data['state'][k+l, :2] * (96 / 512)
            img2[int(agent_pos_pred[1]), int(agent_pos_pred[0])] = [0, 255/(H+1)*(l), 0]


        img2[int(agent_act[1]), int(agent_act[0])] = [255, 0, 0]


        plt.imshow(img2/255)
        plt.show()




