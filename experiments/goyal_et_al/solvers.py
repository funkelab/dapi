import torch
from torch import nn
from losses import EntropyLoss
from tqdm import tqdm
import einops
from pdb import set_trace
import numpy as np


class BatchedExhaustiveSolver:
    """
    Greedy Sequential Exhaustive Search Version Goyal et al
    """

    def best_edit(
        self,
        query_features,
        distractor_features,
        distractor_label,
        previous_edits,
        model,
    ):

        num_patches = query_features.shape[-1]

        list_of_from_to = []
        permuted_hybrid_features = []

        for copy_to in tqdm(range(num_patches), leave=False):
            hybrid_features = query_features.clone()
            for copy_from in range(num_patches):

                hybrid_features[:, copy_to] = distractor_features[:, copy_from]
                list_of_from_to.append([copy_from, copy_to])
                permuted_hybrid_features.append(
                    einops.rearrange(hybrid_features.clone(), "c n -> () (c n)")
                )

        # b (c n)
        permuted_hybrid_features = torch.cat(permuted_hybrid_features)
        with torch.no_grad():

            y_hat = torch.softmax(model.classifier(permuted_hybrid_features), 1)
        target_prediction = y_hat[:, distractor_label]
        # target_prediction: (b,)

        assert len(target_prediction.shape) == 1
        assert target_prediction.shape[0] == num_patches * num_patches

        best_edit = None
        target_prediction = target_prediction.detach().cpu().numpy()

        l = [m[0] for m in previous_edits]  # from
        r = [m[1] for m in previous_edits]  # to

        # we want to insure that that we are not copying from same place

        while best_edit is None or (best_edit[0] in l or (best_edit[1] in r)):
            best_index = target_prediction.argmax()

            # make sure this is not the max anymore
            target_prediction[best_index] = -1

            best_edit = list_of_from_to[best_index]

        # we have to find best feature prediction
        # for this we will take argmax of y_hat (num, num_classes)
        # and select best index for accordint o while loop rules
        best_feature_prediction = y_hat[best_index].detach().cpu().numpy()
        return best_edit, best_feature_prediction


class ContinuousSolver:
    """
    Implementation of Continuous Relaxation Goyal et al
    """

    def __init__(self, max_num_iterations):

        self.max_num_iterations = max_num_iterations
        tqdm.write(
            "Created new ContinuousSolver, " f"max_num_iterations={max_num_iterations}"
        )

    def best_edit(
        self,
        query_features,
        distractor_features,
        distractor_label,
        previous_edits,
        model,
    ):

        self.optimization_problem = ContinuousProblem(
            query_features, distractor_features
        ).cuda()

        tqdm.write(f"Sum of query_features: {query_features.sum()}")

        optim = torch.optim.Adam(self.optimization_problem.parameters(), lr=0.3)
        loss_class = nn.CrossEntropyLoss()
        loss_entropy = EntropyLoss()
        target = distractor_label

        for i in tqdm(range(self.max_num_iterations), leave=False):

            solution = self.optimization_problem(model)

            class_loss = loss_class(
                solution, torch.tensor([target], dtype=torch.long).cuda()
            )
            entropy_loss = loss_entropy(
                self.optimization_problem.a, self.optimization_problem.m
            )
            total_loss = (class_loss + entropy_loss) / 2

            # if converged, we can stop early
            if solution.argmax().item() == target:
                tqdm.write(f"Features classified as {target}, stopping optimization")
                break

            total_loss.backward()
            optim.step()
            optim.zero_grad()

        # ----------- optimization finished ----------
        tqdm.write("Optimzation finished")

        # read out edit (from, to)

        p = self.optimization_problem.get_p().detach().cpu().numpy()
        a = self.optimization_problem.get_a().detach().cpu().numpy()

        if len(previous_edits) > 0:
            r = [m[1] for m in previous_edits]  # to
            l = [m[0] for m in previous_edits]  # from
            # bassicly if the values was already in previos edit we are
            # assigning =-100 to vector a (which has a sice of n)
            # and asining =-100 to a rows m (which has a size of n, n)
            a[r] = -100
            p[l] = -100

        copy_to = np.argmax(a)
        copy_from = np.argmax(a @ p)
        edit = (int(copy_from), int(copy_to))

        q_f = query_features.clone()
        d_f = distractor_features.clone()
        q_f[:, copy_to] = d_f[:, copy_from]
        with torch.no_grad():
            best_feature_prediction = model.classifier(
                einops.rearrange(q_f, "c n -> () (c n)")
            )
            best_feature_prediction = (
                torch.softmax(best_feature_prediction, -1)
                .view(-1)
                .detach()
                .cpu()
                .numpy()
            )
        return edit, best_feature_prediction


class ContinuousProblem(nn.Module):
    def __init__(self, f_q, f_d):

        super().__init__()
        self.f_q = f_q
        self.f_d = f_d
        _, self.n = self.f_q.shape

        # torch.manual_seed(42)
        self.m = nn.Parameter(torch.rand(self.n, self.n))
        self.a = nn.Parameter(torch.rand(self.n))

    def forward(self, model):

        p = self.get_p()
        a = self.get_a()

        # Distractor image features f (I ) are first rearranged with a
        # permutation matrix P and then selectively replace entries in f(I)
        # according to a binary gating vector a
        x_c = self.f_d @ p
        self.x_h = torch.lerp(self.f_q, x_c, a)
        x = model.classifier(einops.rearrange(self.x_h, "c n -> () (c n)"))
        return x

    def get_p(self):
        return torch.softmax(self.m, dim=1)

    def get_a(self):
        return torch.softmax(self.a, dim=-1)