import os
import gin
import torch
import logging
import numpy as np
import soundfile as sf
from tqdm import tqdm
from typing import Tuple, Optional
from pytorch_lightning import seed_everything
from sklearn.neighbors import KNeighborsClassifier

from vggish_utils import vggish_preprocess


@gin.configurable
class EPrior:
    def __init__(
        self,
        num_iteration: int,
        save_output_per_n_iteration: int,
        container_size: int,
        new_breed_size: int,
        rave_path: str,
        rave_version: str,
        device: str,
        sample_rate: int,
        # classifier_path: str,
        output_path: str,
        latent_dim: Tuple[int, int],
        mutate_methods: Tuple,
        container_init_method: str,
        knn_n_neighbors: int,
        select_method: str,
        random_seed: int = 42,
    ):
        self.device = device
        self.rave_path = rave_path
        self.rave_version = rave_version
        # self.classfier_path = classifier_path
        self.output_path = output_path
        self.sample_rate = sample_rate

        self.container_size = container_size
        self.container_init_method = container_init_method
        self.new_breed_size = new_breed_size

        self.latent_dim = latent_dim
        self.num_iteration = num_iteration
        self.random_seed = random_seed
        self.save_output_per_n_iteration = save_output_per_n_iteration
        self.select_method = select_method
        self.knn_n_neighbors = knn_n_neighbors

        self._valid_check()

        self._init()

    def _valid_check(self):
        valid_container_init_method = ["random"]
        if self.container_init_method not in valid_container_init_method:
            raise ValueError(
                f"Invalid container init method: {self.container_init_method}, supported methods: {valid_container_init_method} "
            )

    def _init(self):
        seed_everything(self.random_seed)

        logging.info("Initializing EPrior")
        ## load rave
        self.rave = torch.jit.load(self.rave_path).to(self.device)
        logging.info("RAVE loaded")

        ## load classifier
        self.vggish = torch.hub.load(
            "harritaylor/torchvggish",
            "vggish",
            postprocess=False,
            device=self.device,
        )
        self.vggish.eval()
        logging.info("VGGish loaded")
        # overwrite preprocess function to allow batch processing
        # self.vggish._preprocess = vggish_preprocess

        ## init container
        if self.container_init_method == "normal":
            self.container = torch.normal(
                mean=0.0,
                std=1.0,
                size=(self.container_size, *self.latent_dim),
            )
        elif self.container_init_method == "random":
            self.container = torch.rand(
                size=(self.container_size, *self.latent_dim),
            )

        ## extra inits
        ### init knn for novelty selection
        if self.select_method == "novelty":
            self.knn = KNeighborsClassifier(n_neighbors=self.knn_n_neighbors)

    def mutate(self):
        """
        Cross over and mutate genes in container.
        """
        new_genes = torch.zeros(self.new_breed_size, *self.latent_dim)

        ### crossover on time axis
        first_parent = torch.arange(self.container_size)
        first_parent = first_parent[torch.randperm(self.container_size)][
            : self.new_breed_size
        ]
        # crossover point from idx 1 to idx length-2
        crossover_point = torch.randint(low=1, high=self.latent_dim[1] - 2, size=(1,))
        second_parent = torch.arange(self.container_size)
        second_parent = second_parent[torch.randperm(self.container_size)][
            : self.new_breed_size
        ]
        # first half from first parent, second half from second parent
        new_genes[:, :, :crossover_point] = self.container[
            first_parent, :, :crossover_point
        ]
        new_genes[:, :, crossover_point:] = self.container[
            second_parent, :, crossover_point:
        ]

        ### mutate one column of each gene
        for gene in new_genes:
            col_idx = torch.randint(low=0, high=self.latent_dim[1] - 1, size=(1,))
            gene[:, col_idx] = torch.normal(
                mean=0.0, std=1.0, size=(self.latent_dim[0], 1)
            )

        return new_genes

    def generate(self, genes: torch.Tensor):
        """
        Render embeddings with RAVE decoder.
        """
        with torch.no_grad():
            outputs = self.rave.decode(genes)  # genes shape: batch, dim, length

        # RAVE V1 has output shape (1, batch, time), V2 has shape (batch, 1, time)
        if self.rave_version == "V1":
            outputs = outputs[0]
        elif self.rave_version == "V2":
            outputs = outputs[:, 0, :]

        # final shape: (batch, time)
        return outputs

    def save_results(
        self,
        genes: torch.Tensor,
        output: torch.Tensor,
        out_path: str,
        select_meta: Optional[dict] = None,
    ):
        os.makedirs(out_path)
        # save new genes
        if genes is not None:
            np.save(os.path.join(out_path, "genes.npy"), genes)
        # save select meta
        if select_meta:
            for k in select_meta:
                np.save(os.path.join(out_path, f"{k}.npy"), select_meta[k])
        # save audio files
        for i, out in enumerate(output):
            sf.write(
                out_path + f"/{i:03}.wav",
                out.cpu().numpy(),
                samplerate=self.sample_rate,
            )

    def evaluate(
        self,
        audios: torch.Tensor,
    ):
        embeddings = torch.zeros(audios.shape[0], 128)
        for i, audio in enumerate(audios):
            embedding = self.vggish.forward(audio.cpu().numpy(), self.sample_rate)
            # VGGish outputs more than one vectors when audio is too long, here we take the mean
            if len(embedding.shape) > 1:
                embeddings[i] = embedding.mean(axis=0)
            else:
                embeddings[i] = embedding
        return embeddings

    def select(
        self,
        new_genes: torch.Tensor,
        embeddings: torch.Tensor,
    ):
        if self.select_method == "concat":
            self.container = torch.concat([self.container, new_genes], dim=0)
        elif self.select_method == "replace":
            self.container = new_genes
        elif self.select_method == "novelty":
            ### perform novelty search using knn
            ### perform a knn, and calculate the mean distance for each sample to its neighbors
            ### then select the samples with the highest mean distance
            all_embeddings = torch.concat(
                [self.evaluate(self.generate(self.container)), embeddings], dim=0
            )
            all_genes = torch.concat([self.container, new_genes], dim=0)
            mean_distances = torch.zeros(all_embeddings.shape[0])
            for i, embedding in enumerate(all_embeddings):
                dist = torch.norm(all_embeddings - embedding, dim=1)
                mean_distances[i] = torch.topk(
                    dist, self.knn_n_neighbors, largest=True
                ).values.mean()
            ### update the container with the samples with the highest mean distance
            self.container = all_genes[
                torch.topk(mean_distances, self.container_size, largest=True).indices
            ]

            return {"mean_distances": mean_distances.detach().cpu().numpy()}

        # TODO: add QD selection
        else:
            raise NotImplementedError(
                f"Select method {self.select_method} not implemented"
            )

    def run(self):
        # save init container
        output = self.generate(self.container)

        out_path = os.path.join(self.output_path, f"{0:04}")
        self.save_results(self.container, output, out_path)

        logging.info("Start evolution")
        for n_iter in tqdm(range(self.num_iteration)):
            new_genes = self.mutate()

            output = self.generate(new_genes)

            # evaluate output with VGGish
            embeddings = self.evaluate(output)

            select_meta = self.select(new_genes, embeddings)

            # save output every n iteration, and the init
            if (self.save_output_per_n_iteration > 0) & (
                n_iter % self.save_output_per_n_iteration == 0
            ):
                out_path = os.path.join(self.output_path, f"{n_iter+1:04}")
                self.save_results(
                    self.container,
                    self.generate(self.container),
                    out_path,
                    select_meta=select_meta,
                )

        logging.info("Evolution finished!!")
