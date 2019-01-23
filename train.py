import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model
from collections import deque

def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    # TODO: CODE BEGIN
    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    model = Model().cuda()
    model.load("./checkpoints/model-201812251857-171000.pth")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    num_steps_to_display = 200
    num_steps_to_snapshot = 1000
    num_steps_to_finish = 30000000
    
    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    #raise NotImplementedError
    # TODO: CODE END

    print('Start training')

    while not should_stop:
        # TODO: CODE BEGIN
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            labels = labels.cuda()

            # TODO: CODE BEGIN
            output = model(images)
            loss = model.loss(output, labels)
            # TODO: CODE END

            # TODO: CODE BEGIN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TODO: CODE END

            losses.append(loss.item())
            step += 1

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f} ({steps_per_sec:.2f} steps/sec)')

            if step % num_steps_to_snapshot == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')

            if step == num_steps_to_finish:
                should_stop = True
                break
        #raise NotImplementedError
        # TODO: CODE END

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
