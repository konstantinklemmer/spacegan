import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import display, clear_output


class SpaceGAN:
    def __init__(self, prob_config, check_config, disc_method, gen_method):
        # problem dimension
        self.latent = prob_config["noise_type"]  # noise function
        self.latent_dim = prob_config["noise_dim"]  # noise size
        self.latent_params = prob_config["noise_params"]  # noise params
        self.conditional_dim = prob_config["cond_dim"]  # conditional info
        self.output_dim = prob_config["output_dim"] # output dim
        self.output_labels = prob_config["output_labels"]
        self.input_labels = prob_config["input_labels"]

        # training
        self.epochs = prob_config["epochs"]
        self.batch_size = prob_config["batch_size"]
        self.discriminator = disc_method
        self.disc_opt = prob_config["disc_opt"](self.discriminator.parameters(), **prob_config["disc_opt_params"])
        self.generator = gen_method
        self.gen_opt = prob_config["gen_opt"](self.generator.parameters(), **prob_config["gen_opt_params"])
        self.loss = prob_config["adversarial_loss"]
        self.device = prob_config["device"]
        self.print_results = prob_config["print_results"]

        # selection
        self.check_interval = check_config["check_interval"]
        self.generate_image = check_config["generate_image"]
        self.n_samples = check_config["n_samples"]
        self.perf_metrics = check_config["perf_metrics"]
        self.perf_metrics_setting = check_config["pf_metrics_setting"]
        self.agg_funcs = check_config["agg_funcs"]
        self.agg_metrics = check_config["agg_metrics"]

        # scaling for input and output - a sklearn scaler class
        self.scaling_method_x = prob_config["scale_x"]
        self.scaling_method_y = prob_config["scale_y"]

        # plotting
        if self.print_results & self.generate_image:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        elif self.print_results:
            self.fig, self.ax1 = plt.subplots(1, 1, figsize=(7, 5))
        elif self.generate_image:
            self.fig, self.ax2 = plt.subplots(1, 1, figsize=(7, 5))
        else:
            self.fig = None

    def fit_scaling_method(self, x=None, y=None):
        if x is not None:
            self.scaling_method_x.fit(x)

        if y is not None:
            self.scaling_method_y.fit(y)

    def train(self, x_train, y_train, coords):
        # adversarial ground truths
        valid = torch.ones((self.batch_size, 1)).to(self.device).float()
        fake = torch.zeros((self.batch_size, 1)).to(self.device).float()
        self.df_losses = pd.DataFrame(index=range(self.epochs), columns=["D", "G"])

        # processed input and output
        self.fit_scaling_method(x_train, y_train)  # scaling procedure
        x_train = self.scaling_method_x.transform(x_train)
        y_train = self.scaling_method_y.transform(y_train)

        # tensors
        x_train = torch.from_numpy(x_train).to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)

        for epoch in range(self.epochs):
            # ------------
            # Get minibatch
            # ------------
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            true_seq = y_train[idx]
            noise = self.latent(size=(self.batch_size, self.latent_dim), params=self.latent_params)
            noise = torch.from_numpy(noise).to(self.device)

            # ------------
            # Train Generator
            # ------------
            self.generator.zero_grad()

            # generate a batch of new data
            gen_seq = self.generator(noise, x_train[idx]).double()

            # train the generator (to have the discriminator label samples as valid)
            g_loss = self.loss(self.discriminator(gen_seq, x_train[idx]), valid)
            g_loss.backward()
            self.gen_opt.step()

            # -------------
            # Train Discriminator
            # -------------
            self.discriminator.zero_grad()

            # train the discriminator
            d_loss_real = self.loss(self.discriminator(true_seq, x_train[idx]), valid)
            d_loss_fake = self.loss(self.discriminator(gen_seq.detach(), x_train[idx]), fake)
            d_loss = 0.5 * torch.add(d_loss_real, d_loss_fake)

            d_loss.backward()
            self.disc_opt.step()

            # store losses
            self.df_losses.loc[epoch, ["D"]] = d_loss.item()
            self.df_losses.loc[epoch, ["G"]] = g_loss.item()

            # plot the progress
            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, self.epochs, d_loss.item(), g_loss.item()), end="\r")

            # get images
            if self.print_results:
                self.ax1.clear()
                self.df_losses.plot(ax=self.ax1)
                self.ax1.set_xlim(0, epoch+1)

            # if at save interval => save generated output samples
            if self.check_interval is not None:
                if epoch % self.check_interval == 0:
                    self.checkpoint_model(epoch)
                    if self.generate_image:
                        self.print_image(epoch, x_train, coords)
            if self.fig is not None:
                clear_output(wait=True)
                display(self.fig)

    def predict(self, x, noise=None):
        # generate noise
        x = torch.from_numpy(self.scaling_method_x.transform(x)).to(self.device)
        if noise is None:
            noise = self.latent(size=(x.shape[0], self.latent_dim), params=self.latent_params)
            noise = torch.from_numpy(noise).to(self.device)

        # generate sequence
        gen_seq = self.generator(noise, x)
        return self.scaling_method_y.inverse_transform(gen_seq.detach().cpu().numpy())

    def checkpoint_model(self, epoch):
        # generator
        torch.save(self.generator, "gen_iter %d.pkl.gz" % epoch)

        # discriminator
        torch.save(self.discriminator, "disc_iter %d.pkl.gz" % epoch)

    def print_image(self, epoch, x_train, coords):

        # generate noise
        noise = self.latent(size=(x_train.size(0), self.latent_dim), params=self.latent_params)
        noise = torch.from_numpy(noise).to(self.device)

        # generate predictions
        gen_seq = self.generator(noise, x_train)
        for _ in range(1, self.n_samples):
            # generate noise
            noise = self.latent(size=(x_train.size(0), self.latent_dim), params=self.latent_params)
            noise = torch.from_numpy(noise).to(self.device)

            # predictions
            gen_seq += self.generator(noise, x_train)
        gen_seq /= self.n_samples

        # create charts
        for (i, label) in enumerate(self.output_labels):
            # charts
            norm_gan_mean = (gen_seq[:, i] - min(gen_seq[:, i])) / (max(gen_seq[:, i]) - min(gen_seq[:, i]))
            colors = cm.rainbow(norm_gan_mean.detach().cpu().numpy())

            # plotting
            self.ax2.clear()
            for lat, long, c in zip(coords[:, 0], coords[:, 1], colors):
                self.ax2.scatter(lat, long, color=c)
            self.ax2.set_xlabel(r'$c^{(1)}$', fontsize=14)
            self.ax2.set_ylabel(r'$c^{(2)}$', fontsize=14)
            self.ax2.set_title("SpaceGAN - Epoch " + str(epoch) + " - Variable " + label)
            # self.fig.savefig(label + "_iter_" + str(epoch) + ".png", dpi = 100, transparent=True, bbox_inches="tight")
