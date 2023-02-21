"""
This script is modified from Time-series Generative Adversarial Networks (TimeGAN) Codebase
Link: https://github.com/jsyoon0823/TimeGAN

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
import argparse

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def visualization(model_name, dataset, ori_data, generated_data, analysis, max_sample_size=1000):
    """Using PCA or tSNE for generated and original data visualization.

  Args:
    - ori_data: original data (shape=[num_samples, seq_len, num_sensors])
    - generated_data: generated synthetic data (shape=[num_samples, seq_len, num_sensors])
    - analysis: tsne or pca
    - max_sample_size: max sample size (default=1000)
  """
    print("shape of ori_data:", ori_data.shape)
    print("shape of generated_data:", generated_data.shape)

    # Analysis sample size (for faster computation)
    anal_sample_no = min([max_sample_size, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    # 对sensor取平均值
    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':

        # Do PCA Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # PCA Analysis
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(prep_data_final)

        # Plotting
        plt.figure()
        plt.scatter(pca_results[:anal_sample_no, 0], pca_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_results[anal_sample_no:, 0], pca_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        plt.legend()
        plt.title(f'{dataset} PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.savefig("./{}_{}_pca.png".format(model_name, dataset), bbox_inches="tight", dpi=300)
        plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        plt.figure()
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        plt.legend()
        plt.title(f'{dataset} t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig("./{}_{}_tsne.png".format(model_name, dataset), bbox_inches="tight", dpi=300)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Time Series Generation")
    parser.add_argument(
        "--model_name",
        choices=['TimeGAE', 'TimeVAE', 'TimeGAN', "TTSGAN"],
        default='TimeGAE',
        type=str
    )
    parser.add_argument(
        "--dataset",
        choices=["golf", "energy", "winnipeg"],
        default="golf",
        type=str
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.dataset == "golf":
        ori_data = np.load("../../golf_dataset/olddata/old_data_preprocess_no_aug.npy").swapaxes(1, 2)  # 213 645 8
    elif args.dataset == "energy":
        ori_data = np.load("../../energyDataset/energy.npy").swapaxes(1, 2)
    else:
        ori_data = np.load("../../WinnipegDataset/winnipeg_1.npy").swapaxes(1, 2)

    if args.model_name == "TimeGAN":
        generated_data = np.load(f"../TimeGAN/timegan_gen_{args.dataset}.npy")
    elif args.model_name == "TimeGAE":
        generated_data = np.load(f"../TimeGAE/timegae_gen_{args.dataset}_transformer.npy")
    elif args.model_name == "TimeGVAE":
        generated_data = np.load(f"../TimeGAE/timegvae_gen_{args.dataset}_transformer.npy")
    elif args.model_name == "TimeVAE":
        generated_data = np.load(f"../TimeVAE/outputs/timevae_gen_{args.dataset}.npy")
    else:
        generated_data = np.load(f"../TTSGAN/ttsgan_gen_{args.dataset}.npy").swapaxes(1, 2)

    visualization(model_name=args.model_name, dataset=args.dataset, ori_data=ori_data, generated_data=generated_data,
                  analysis="pca")
    visualization(model_name=args.model_name, dataset=args.dataset, ori_data=ori_data, generated_data=generated_data,
                  analysis="tsne")
