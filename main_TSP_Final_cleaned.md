KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics

State estimation of dynamical systems in RT is a fundamental task in signal processing. For systems that are well-represented by a fully known LG SS model, the celebrated KF is a low complexity optimal solution. However, both linearity of the underlying SS model and accurate knowledge of it are often not encountered in practice. Here, we present KN, a RT state estimator that learns from data to carry out Kalman filtering under non-linear dynamics with partial information. By incorporating the structural SS model with a dedicated RNN module in the flow of the KF, we retain data efficiency and interpretability of the classic algorithm while implicitly learning complex dynamics from data. We demonstrate numerically that KN overcomes non-linearities and model mismatch, outperforming classic filtering methods operating with both mismatched and accurate domain knowledge.

# Introduction
Estimating the hidden state of a dynamical system from noisy observations in RT is one of the most fundamental tasks in signal processing and control, with applications in localization, tracking, and navigation. In a pioneering work from the early 1960s, based on work by {Wiener from 1949, Rudolf Kalman introduced the KF, a MMSE estimator that is applicable to time-varying systems in DT, which are characterized by a linear SS model with AWGN.}
The low-complexity implementation of the KF, combined with its sound theoretical basis, resulted in it quickly becoming the leading workhorse of SE in systems that are well described by SS models in DT. The KF has been applied to problems such as radar target tracking, trajectory estimation of ballistic missiles, and estimating the position and velocity of a space vehicle in the Apollo program.

While the original KF assumes linear SS models, many problems encountered in practice are governed by NL dynamical equations. Therefore, shortly after the introduction of the original KF, NL variations of it were proposed, such as the EKF and the UKF. Methods based on sequential MC sampling, such as the family of PF, were introduced for state estimation in NL, non-Gaussian SS models.
To date, the KF and its NL variants are still widely used for online filtering in numerous RW applications involving tracking and localization.

The common thread among these aforementioned filters is that they are MB algorithms; namely, they rely on accurate
knowledge and modeling of the underlying dynamics as a fully characterized SS model. As such, the performance of these MB methods critically depends on the validity of the domain knowledge and model assumptions. MB filtering algorithms designed to cope with some level of uncertainty in the SS models, e.g.,, are rarely capable of achieving the performance of MB filtering with full domain knowledge, and rely on some knowledge of how much their postulated model deviates from the true one. In many practical use cases the underlying dynamics of the system is NL, complex, and difficult to accurately characterize as a tractable SS model, in which case degradation in performance of the MB state estimators is expected.

Recent years have witnessed remarkable empirical success of DNN in real-life applications. These DD parametric models were shown to be able to catch the subtleties of complex processes and replace the need to explicitly characterize the domain of interest. Therefore, an alternative strategy to implement SE—without requiring explicit and accurate knowledge of the SS model—is to learn this task from data using deep learning. DNN such as RNN—i.e., LSTM and GRU—and attention mechanisms have been shown to perform very well for time series related tasks mostly in intractable environments, by training these networks in an E2E, model-agnostic manner from a large quantity of data. Nonetheless, DNN do not incorporate domain knowledge such as structured SS models in a principled manner. Consequently, these DD approaches require many trainable parameters and large data sets even for simple sequences and lack the interpretability of MB methods. These constraints limit the use of highly parametrized DNN for RT SE in applications embedded in hardware-limited mobile devices such as drones and vehicular systems.

The limitations of MB Kalman filtering and DD SE motivate a hybrid approach that exploits the best of both worlds; i.e., the soundness and low complexity of the classic KF, and the model-agnostic nature of DNN. Therefore, we build upon the success of our previous work in MB DL for signal processing and digital communication applications to propose a hybrid MB/DD online recursive filter, coined KN. In particular, we focus on RT SE for continuous-value SS models for which the KF and its variants are designed. We assume that the noise statistics are unknown and the underlying SS model is partially known or approximated from a physical model of the system dynamics. To design KN, we identify the KG computation of the KF as a critical component encapsulating the dependency on noise statistics and domain knowledge, and replace it with a compact RNN of limited complexity that is integrated into the KF flow. The resulting system uses labeled data to learn to carry out Kalman filtering in a supervised manner.

Our main contributions are summarized as follows:
- We design KN, which is an interpretable, low complexity, and data-efficient DNN-aided RT state estimator. KN builds upon the flow and theoretical principles of the KF, incorporating partial domain knowledge of the underlying SS model in its operation.
- By learning the KG, KN circumvents the dependency of the KF on knowledge of the underlying noise statistics, thus bypassing numerically problematic matrix inversions involved in the KF equations and overcoming the need for tailored solutions for NL systems; e.g., approximations to handle non-linearities as in the EKF.
- {We show that KN learns to carry out KFING from data in a manner that is invariant to the sequence length. Specifically, we present an efficient supervised training scheme that enables KN to operate with arbitrary long trajectories while only training using short trajectories.}
- We evaluate KN in various SS models. The experimental scenarios include synthetic setups, tracking the chaotic Lorenz system, and localization using the Michigan NCLT DS. KN is shown to converge much faster compared with purely DD systems, while outperforming the MB EKF, UKF, and PF, when facing model mismatch and dominant non-linearities.
The proposed KN leverages data and partial domain knowledge to learn the filtering operation, rather than using data to explicitly estimate the missing SS model parameters. Although there is a large body of work that combines SS models with DNN, e.g.,, these approaches are sometimes used for different SS related tasks; with a different focus, e.g., incorporating high-dimensional visual observations to a KF; or under different assumptions, as we discuss in detail below.

The rest of this paper is organized as follows:
Section~ reviews the SS model and its associated tasks, and discusses related works. Section~ details the proposed KN. Section~ presents the numerical study. Section~ provides concluding remarks and future work.

Throughout the paper, we use boldface lower-case letters for vectors and boldface upper-case letters for matrices. The transpose, $\ell_2$ norm, and stochastic expectation are denoted by $\set{*}^\top$, $\norm{*}$, and $\expecteds{*}$, respectively. The Gaussian distribution with mean $\mu$ and covariance $\Sigma$ is denoted by $\mathcal{N}(\mu, \Sigma)$. Finally, $\greal$ and $\gint$ are the sets of real and integer numbers, respectively.

# System Model and Preliminaries
## State Space Model
We consider dynamical systems characterized by a SS model in DT.
We focus on NL, Gaussian, and continuous SS models, which for each $t\in\gint$ are represented via
x_t=
f\brackets{x_t-1}+w_t,
w_t\sim
\mathcal{N}\brackets{0,Q},

x_t\in\greal^m,
y_t=
h\brackets{x_t}+v_t,
v_t\sim
\mathcal{N}\brackets{0,R},

y_t\in\greal^n.
In, $x_t$ is the latent state vector of the system at time t, which evolves from the previous state $x_t-1$, by a NL, state-evolution function $f\brackets{*}$ and by an AWGN $w_t$ with covariance matrix $Q$. In, $y_t$ is the vector of observations at time t, which is generated from the current latent state vector by a NL observation mapping $h\brackets{*}$ corrupted byAWGN $v_t$ with covariance $R$. For the special case where the evolution or the observation transformations are linear, there exist matrices $F,H$ such that
f\brackets{x_t-1}={F}*x_t-1,

h\brackets{x_t}={H}*x_t.

In practice, the state-evolution model is determined by the complex dynamics of the underlying system, while the observation model is dictated by the type and quality of the observations. For instance, $x_t$ can determine the location, velocity, and acceleration of a vehicle, while $y_t$ are measurements obtained from several sensors. The parameters of these models may be unknown and often require the introduction of dedicated mechanisms for their estimation in real-time. In some scenarios, one is likely to have access to an approximated or mismatched characterization of the underlying dynamics.

SS models are studied
in the context of several different tasks; these tasks are different in their nature, and can be roughly classified into two main categories: observation approximation and hidden state recovery.
The first category deals with approximating parts of the observed signal $y_t$. This can correspond, for example, to the prediction of future observations given past observations; the generation of missing observations in a given block via imputation; and the denoising of the observations. The second category considers the recovery of a hidden state vector $x_t$. This family of state recovery tasks includes offline recovery, also referred to as smoothing, where one must recover a block of hidden state vectors, given a block of observations, e.g.,. The focus of this paper is filtering; i.e., online recovery of $x_t$ from past and current noisy observations $\{y_\tau\}_\tau =1^t$. For a given $x_0$, filtering involves the design of a mapping from $y_t$ to $\hat{x}_t$, $\forall t\in\set{1,2,\ldots,T} \triangleq \mathcal{T}$, where T is the time horizon.

## Data-Aided Filtering Problem Formulation
The filtering problem is at the core of RT tracking. Here, one must provide an instantaneous estimate of the state ${x}_t$ based on each incoming observation $y_t$ in an online manner.
Our main focus is on scenarios where one has partial knowledge of the SS model that describes the underlying dynamics. Namely, we know the state-evolution function $\gevol$ and the state-observation function $\gobs$. For RW applications, this knowledge is derived from our understating of the system dynamics, its physical design, and the model of the sensors. As opposed to the classical assumptions in KF, the noise statistics $Q$ and $R$ are not known. More specifically, we assume:
- Knowledge of the distribution of the noise signals $w_t$ and $v_t$ is not available.
- The functions $\gevol$ and $\gobs$ may constitute an approximation of the true underlying dynamics. Such approximations can correspond, for instance, to the representation of continuous time dynamics in discrete time, acquisition using misaligned sensors, and other forms of mismatches.

While we focus on filtering in partially known SS models, we assume that we have access to a labeled DS containing a sequence of observations and their corresponding GT states. In various scenarios of interest, one can assume access to some GT measurements in the design stage. For example, in field experiments it is possible to add extra sensors both internally or externally to collect the GT needed for training.
It is also possible to compute the GT data using offline and more computationally intensive algorithms.
Finally, the inference complexity of the learned filter should be of the same order as that of MB filters, such as the EKF.

## Related Work
A key ingredient in recursive Bayesian filtering is the update operation; namely, the need to {update} the prior estimate using new observed information. For LG SS using the KF, this boils down to computing the KG. While the KF assumes linear SS models, many problems encountered in practice are governed by NL dynamics, for which one should resort to approximations.
Several extensions of the KF were proposed to deal with non-linearities. The EKF is a quasi-linear algorithm based on an analytical linearization of the SS model. More recent NL variations are based on numerical integration: UKF, the Gauss-Hermite Quadrature, and the Cubature KF.
For more complex SS models, and when the noise cannot be modeled as Gaussian, multiple variants of the PF were proposed that are based on sequential MC. These MC algorithms are considered to be asymptotically exact but relatively computationally heavy when compared to Kalman-based algorithms. These MB algorithms require accurate knowledge of the SS model, and their performance is typically degrades in the presence of model mismatch.

The combination of ML and SS models, and specifically {Kalman}-based algorithms, is the focus of growing research attention. To frame the current work in the context of existing literature, we focus on the approaches that preserve the general structure of the SS model. The conventional approach to deal with partially known SS models is to impose a parametric model and then estimate its parameters. This can be achieved by {jointly} learning the parameters and state sequence using EM and Bayesian probabilistic algorithms, or by selecting from a set of a priori known models. When training data is available, it is commonly used to tune the missing parameters in advance, in a supervised or an unsupervised manner, as done in. The main drawback of these strategies is that they are restricted to an imposed parametric model on the underlying dynamics.

When one can bound the uncertainty in the SS model in advance, an alternative approach to learning is to minimize the {worst-case} estimation error among all expected SS models. Such robust variations were proposed for various state estimation algorithms, including Kalman variants and particle filters. The fact that these approaches aim to design the filter to be suitable for multiple different SS models typically results in degraded performance compared to operating with known dynamics.

When the underlying system's dynamics are complex and only partially known or the emission model is intractable and cannot be captured in a closed form—e.g., visual observations as in a computer vision task—one can resort to approximations and to the use of DNN. Variational inference is commonly used in connection with SS models, as in, by casting the Bayesian inference task to optimization of a parameterized posterior and maximizing an objective. Such approaches cannot typically be applied directly to state recovery in RT, as we consider here, and the learning procedure tends to be complex and prone to approximation errors.

A common strategy when using DNN is to encode the observations into some latent space that is assumed to obey a simple SS model, typically a linear Gaussian one, and track the state in the latent domain as in, or to use DNN to estimate the parameters of the SS model as in. Tracking in the latent space can also be extended by applying a DNN decoder to the estimated state to return to the observations domain, while training the overall system end-to-end. The latter allows to design trainable systems for recovering missing observations and predicting future ones by assuming that the temporal relationship can be captured as an SS model in the latent space. This form of DNN-aided systems is typically designed for unknown or highly complex SS models, while we focus in this work on setups with partial domain knowledge, as detailed in Subsection~.
Another approach is to combine RNN, or variational inference with MC based sampling.
Also related is the work, which used learned models in parallel with MB algorithms operating with full knowledge of the SS model, applying a graph neural network in parallel to the Kalman smoother to improve its accuracy via neural augmentation. Estimation was performed by an iterative message passing over the entire time horizon. This approach is suitable for the smoothing task and is computationally intensive, and so may not be suitable for RT filtering.
## Model-Based Kalman Filtering
Our proposed KN, detailed in the following section, is based on the MB KF, which is a linear recursive estimator. In every time step t, the KF produces a new estimate $x_t$ using only the previous estimate $\hat{x}_t-1$ as a sufficient statistic and the new observation $y_t$. As a result, the computational complexity of the KF does not grow in time. We first describe the original algorithm for linear SS models, as in, and then discuss how it is extended into the EKF for NL SS models.

The KF can be described by a two-step procedure: prediction and update, where in each time step $t\in \mathcal{T}$, it computes the first- and second-order statistical moments.
- The first step predicts the current a priori statistical moments based on the previous a posteriori estimates. Specifically, the moments of $x$ are computed using the knowledge of the evolution matrix $F$ as
\hat{x}_{t\given{t-1}} =
F*{\hat{x}_{t-1\given{t-1}}},
\mySigma_{t\given{t-1}} =
{F}*\mySigma_{t-1\given{t-1}}*{F}^\top+Q
and the moments of the observations $y$ are computed based on the knowledge of the observation matrix $H$ as
\hat{y}_{t\given{t-1}} =
H*\hat{x}_{t\given{t-1}}
{S}_{t\given{t-1}} =
{H}*\mySigma_{t\given{t-1}}*{H}^\top+R.
- In the update step, the a posteriori state moments are computed based on the a priori moments as
\hat{x}_{t\given{t}}=
\hat{x}_{t\given{t-1}}+\Kgain_t*\Deltay_t
{\mySigma}_{t\given{t}}=
{\mySigma}_{t\given{t-1}}-\Kgain_t*{\mathbf{S}}_{t\given{t-1}}*\Kgain^\top_t.
Here, $\Kgain_t$ is the KG, and it is given by
\Kgain_t={\mySigma}_{t\given{t-1}}*{H}^\top*{S}^-1_{t\given{t-1}}.
The term $\Deltay_t$ is the innovation; i.e., the difference between the predicted observation and the observed value, and it is the only term that depends on the observed data
\Deltay_t=y_t-\hat{y}_{t\given{t-1}}.

The EKF extends the KF for NL $f\brackets{*}$ and/or $h\brackets{*}$, as in. Here, the first-order statistical moments and are replaced with
\hat{x}_{t\given{t-1}} =
f\brackets{\hat{x}_t-1},
\hat{y}_{t\given{t-1}} =
h\brackets{\hat{x}_{t\given{t-1}}},
respectively. The second-order moments, though, cannot be propagated through the non-linearity, and must thus be approximated.
The EKF linearizes the differentiable $f\brackets{*}$ and $h\brackets{*}$ in a time-dependent manner using their partial derivative matrices, also known as Jacobians, evaluated at $\hat{x}_{t-1\given{t-1}}$ and $\hat{x}_{t\given{t-1}}$. Namely,
\hat{F}_t=\jacob_f\brackets{\hat{x}_{t-1\given{t-1}}}
\hat{H}_t=\jacob_h\brackets{\hat{x}_{t\given{t-1}}},
where $\hat{F}_t$ is plugged into and $\hat{H}_t$ is used in and. When the SS model is linear, the EKF coincides with the KF, which achieves the MMSE for linear Gaussian SS models.

An illustration of the EKF is depicted in Fig.~. The resulting filter admits an efficient linear recursive structure.
However, it requires full knowledge of the underlying model and notably degrades in the presence of model mismatch. When the model is highly NL, the local linearity approximation may not hold, and the EKF can result in degraded performance. This motivates the augmentation of the EKF into the deep learning-aided KN, detailed next.
{Blocks_v1/EKF_Block.pdf}
{EKF block diagram. Here, $\mathcal{Z}^-1$ is the unit delay.}

# KalmanNet

Here, we present KN; a hybrid, interpretable, data efficient architecture for RT SE in NL dynamical systems with partial domain knowledge. KN combines MB Kalman filtering with an RNN to cope with model mismatch and non-linearities. To introduce KN, we begin by explaining its high level operation in Subsection~. Then we present the features processed by its internal RNN and the specific architectures considered for implementing and training KN in Subsections~-. Finally, we provide a discussion in Subsection~.
## High Level Architecture
We formulate KN by identifying the specific computations of the EKF that are based on unavailable knowledge. As detailed in Subsection~, the functions $\gevol$ and $\gobs$ are known; yet the covariance matrices $Q$ and $R$ are unavailable. These missing statistical moments are used in MB Kalman filtering only for computing the KG (see Fig.~). Thus, we design KN to learn the KG from data, and combine the learned KG in the overall KF flow. This high level architecture is illustrated in Fig.~.

In each time instance $t \in \mathcal{T}$, similarly to the EKF, KN estimates $\hat{x}_t$ in two steps; prediction and update.
- The prediction step is the same as in the MB EKF, except that only the first-order statistical moments are predicted. In particular, a prior estimate for the current state $\priorst$ is computed from the previous posterior $\postst{t-1}$ via. Then, a prior estimate for the current observation $\priorobs$ is computed from $\priorst$ via. As opposed to its MB counterparts, KN does not rely on the knowledge of noise distribution and does not maintain an explicit estimate of the COV.
- In the update step, KN uses the new observation $y_t$ to compute the current state posterior $\postst{t}$ from the previously computed prior $\priorst$ in a similar manner to the MB KF as in, i.e., using the innovation term $\ino{t}$ computed via and the KG $\Kgain_t$. As opposed to the MB EKF, here the computation of the KG {is not given} explicitly; rather, it is learned from data using an RNN, as illustrated in Fig.~. The inherent memory of RNN allows to implicitly track the second-order statistical moments without requiring knowledge of the underlying noise statistics.
Designing an RNN to learn how to compute the KG as part of an overall KF flow requires answers to three key questions:
- From which input features will the network learn the KG?
- What should be the architecture of the internal RNN?
- How will this network be trained from data?
In the following sections we address these questions.
{Blocks_v1/KalmanNet_Block.pdf}
{KN block diagram.}

## Input Features
The MB KF and its variants compute the KG from knowledge of the underlying statistics. To implement such computations in a learned fashion, one must provide input that capture the knowledge needed to evaluate the KG to a neural network. The dependence of $\Kgain_t$ on the statistics of the observations and the state process indicates that in order to track it, in every time step $t \in \mathcal{T}$, the RNN should be provided with input containing statistical information of the observations $y_t$ and the state-estimate $\postst{t-1}$. Therefore, the following quantities that are related to the unknown statistical relationship of the SS model can be used as input features to the RNN:
- The observation difference $\Delta\tilde{y}_t={y_t-y_t-1}$.
- The innovation difference $\Deltay_t=y_t-\hat{y}_{t\given{t-1}}$.
- The forward evolution difference $\Delta\tilde{x}_t={\hat{x}_{t\given{t}}-\hat{x}_{{t-1}\given{t-1}}}$. This quantity represents the difference between two consecutive posterior state estimates, where for time instance t, the available feature is $\Delta\tilde{x}_t-1$.
- The forward update difference $\Delta\hat{x}_t={\hat{x}_{t\given{t}}-\hat{x}_{{t}\given{t-1}}}$, i.e., the difference between the posterior state estimate and the prior state estimate, where again for time instance t we use $\Delta\hat{x}_t-1$.

Features and encapsulate information about the state-evolution process, while features and encapsulate the uncertainty of our state estimate. The difference operation removes the predictable components, and thus the time series of differences is mostly affected by the noise statistics that we wish to learn. The RNN described in Fig.~ can use all the features, although extensive empirical evaluation suggests that the specific choice of combination of features depends on the problem at hand. Our empirical observations
indicate that good combinations are \{,,\} and \{,,\}.
## Neural Network Architecture
{Blocks_v1/KNet_DNN_v14.pdf}
{KN RNN block diagram (ARCH1). The architecture comprises a fully connected input layer, followed by a GRU layer (whose internal division into gates is illustrated) and an output fully connected layer. Here, the input features are and.}

The internal DNN of KN uses the features discussed in the previous section to compute the KG.
It follows from that computing the KG $\Kgain_t$ involves tracking the COV ${\mySigma}_t$. The recursive nature of the KG computation indicates that its learned module should involve an internal memory element as an RNN to track it.

We consider two architectures for the KG computing RNN. The first, illustrated in Fig.~, aims at using the internal memory of RNN to jointly track the underlying second-order statistical moments required for computing the KG in an implicit manner. To that aim, we use GRU cells whose hidden state is of the size of some integer product of $m^2 + n^2$, which is the joint dimensionality of the tracked moments $\hat{\Sigma}_t|t-1$ in, and $\hat{S}_t$ in. In particular, we first use a FC input layer whose output is the input to the GRU. The GRU state vector $h_t$ is mapped into the estimated KG $\Kgain_t \in \greal^m* n$ using an output FC layer with $m* n$ neurons. While the illustration in Fig.~ uses a single GRU layer, one can also utilize multiple layers to increase the capacity and abstractness of the network, as we do in the numerical study reported in Subsection~. The proposed architecture does not directly design the hidden state of the GRU to correspond to the unknown second-order statistical moments that are tracked by the MB KF. As such, it uses a relatively large number of state variables that are expected to provide the required tracking capacity. For example, in the numerical study in Section~ we set the dimensionality of $h_t$ to be $10*(m^2 + n^2)$. This often results in substantial over-parameterization, as the number of GRU parameters grows quadratically with the number of state variables.

The second architecture uses separate GRU cells for each of the tracked COV. The division of the architecture into separate GRU cells and FC layers and their interconnection is illustrated in Fig.~. As shown in the figure, the network composes three GRU layers, connected in a cascade with dedicated input and output FC layers. The first GRU layer tracks the unknown state noise covariance $Q$, thus tracking $m^2$ variables. Similarly, the second and third GRU track the predicted moments $\hat{\Sigma}_t|t-1$ and $\hat{S}_t$, thus having $m^2$ and $n^2$ hidden state variables, respectively. The GRU are interconnected such that the learned $Q$ is used to compute $\hat{\Sigma}_t|t-1$, which in turn is used to obtain $\hat{S}_t$, while both $\hat{\Sigma}_t|t-1$ and $\hat{S}_t$ are involved in producing $\Kgain_t$. This architecture, which is composed of a non-standard interconnection between GRU and FC layers, is more directly tailored towards the formulation of the SS model and the operation of the MB KF compared with the simpler first architecture. As such, it provides lesser abstraction; i.e., it is expected to be more constrained in the family of mappings it can learn compared with the first architecture, while as a result also requiring less trainable parameters. For instance, in the numerical study reported in Subsection~, utilizing the first architecture requires the order of $5\cdot10^5$ trainable parameters, while the second architecture utilizes merely $2.5 * 10^4$ parameters.
{Blocks_v1/KNet_DNN_v21.pdf}
{KN RNN block diagram (ARCH2). The input features are used to update three GRU with dedicated FC layers, and the overall interconnection between the blocks is based on the flow of the KG computation in the MB KF.}

## Training Algorithm
KN is trained using the available labeled data set in a supervised manner. While we use a neural network for computing the KG rather than for directly producing the estimate $\hat{x}_t|t$, we train KN end-to-end. Namely, we compute the loss function $\mathcal{L}$ based on the state estimate $\postst{t}$, which is not the output of the internal RNN. Since this vector takes values in a continuous set $\greal^m$, we use the squared-error loss,
\mathcal{L} = \norm{x_t -\hat{x}_t|t}^2
which is also used to evaluate the MB KF. By doing so, we build upon the ability to backpropagate the loss to the computation of the KG. One can obtain the loss gradient with respect to the KG from the output of KN since
\frac{\partial \mathcal{L}}{\partial\Kgain_t} =\frac{\partial \norm{\Kgain_t \ino{t} -\dx{t}}^2}{\partial\Kgain_t } \notag 
=2 * \brackets{\Kgain_t* \ino{t} - \dx{t}}*\ino{t}^\top,

where $\dx{t}\triangleq x_t -\hat{x}_t|t-1$. The gradient computation in indicates that one can learn the computation of the KG by training KN end-to-end using the squared-error loss. In particular, this allows to train the overall filtering system without having to externally provide ground truth values of the KG for training purposes.

The data set used for training comprises N trajectories that can be of varying lengths. Namely, by letting $T_i$ be the length of the ith training trajectory, the data set is given by $\mathcal{D} = \set{(Y_i, X_i)}_1^N$, where
Y_i=[y_1^{\brackets{i}},\ldots,y_T_i^{\brackets{i}}], 
X_i=[x_0^{\brackets{i}},x_1^{\brackets{i}},\ldots,x_T_i^{\brackets{i}}].
By letting $\NNParam$ denote the trainable parameters of the RNN, and $\gamma$ be a regularization coefficient, we then construct an $\ell_2$ regularized MSE loss measure
\ell_i\brackets{\NNParam}=
(1)/(T_i)\sum_t=1^T_i\norm{
\hat{x}_t\brackets{y^{\brackets{i}}_t;\NNParam}\!-\! x^{\brackets{i}}_t}^2+
\gamma*\norm{\NNParam}^2.

 To optimize $\NNParam$, we use a variant of mini-batch SGD in which for every {batch} indexed by k, we choose $M < N$ trajectories indexed by $i_1^k, \ldots, i_M^k$, computing the mini-batch loss as
\mathcal{L}_k\brackets{\NNParam}=
(1)/(M)\sum_j=1^M\ell_i_j^k\brackets{\NNParam}.

Since KN is a recursive architecture with both an external recurrence and an internal RNN, we use the BPTT algorithm to train it. Specifically, we unfold KN across time with shared network parameters, and then compute a forward and backward gradient estimation pass through the network. We consider three different variations of applying the BPTT algorithm for training KN:
- Direct application of BPTT, where for each training iteration the gradients are computed over the entire trajectory.
- An application of the truncated BPTT algorithm. Here, given a DS of long trajectories (e.g., $T=3000$ time steps), each long trajectory is divided into multiple short trajectories (e.g., $T=100$ time steps), which are shuffled and used during training.
- An alternative application of truncated BPTT, where we truncate each trajectory to a fixed length, and train using these short trajectories.

Overall, directly applying BPTT via may be computationally expensive and unstable. Therefore, a favored approach is to first use the truncated BPTT as in as a warm-up phase in order to stabilize its learning process, after which KN is tuned using. The procedure in is most suitable for systems that are known to be likely to quickly converge to a steady state (e.g., linear SS models). In our numerical study, reported in Section~, we utilize all three approaches.
## Discussion
KN is designed to operate in a hybrid DD/MB manner, combining DL with the classical EKF procedure. By identifying the specific noise-model-dependent computations of the EKF and replacing them with a dedicated RNN integrated in the EKF flow, KN benefits from the individual strengths of both DD and MB approaches. The augmentation of the EKF with dedicated deep learning modules results in several core differences between KN and its MB counterpart. Unlike the MB EKF, KN does not attempt to linearize the SS model, and does not impose a statistical model on the noise signals. In addition, KN filters in a non-linear manner, as its KG matrix depends on the input $y_t$. Due to these differences, compared to MB Kalman filtering, KN is more robust to model mismatch and can infer more efficiently, as demonstrated in Section. In particular, the MB EKF is sensitive to inaccuracies in the underlying SS model, e.g., in $\gevol$ and $\gobs$, while KN can overcome such uncertainty by learning an alternative KG that yields accurate estimation.

Furthermore, KN is derived for SS models when noise statistics are not specified explicitly. A MB approach to tackle this without relying on data employs the RKF, which designs the filter to minimize the maximal MSE within some range of assumed SS models, at the cost of performance loss, compared to knowing the true model. When one has access to data, the direct strategy to implement the EKF in such setups is to use the data to estimate $Q$ and $R$, either directly from the data or by backpropagating through the operation of the EKF as in, and utilize these estimates to compute the KG. As covariance estimation can be a challenging task when dealing with high-dimensional signals, KN bypasses this need by directly learning the KG, and by doing so approaches the MSE of MB Kalman filtering with full knowledge of the SS model, as demonstrated in Section~. Finally, the computation complexity
for each time step $t \in \mathcal{T}$ is also linear in the RNN dimensions and does not involve matrix inversion. This implies that KN is a good candidate to apply for high dimensional SS models and on computationally limited devices.

Compared to purely DD state estimation, KN benefits from its model awareness and the fact that its operation follows the flow of MB Kalman filtering rather than being utilized as a black box. As numerically observed in Section~, KN achieves improved MSE compared to utilizing RNN for end-to-end state estimation, and also approaches the MMSE performance achieved by the MB KF in linear Gaussian SS models.
Furthermore, the fact that KN preserves the flow of the EKF implies that the intermediate features exchanged between its modules have a specific operation meaning, providing interpretability that is often scarce in E2E, DL systems. Finally, the fact that KN learns to compute the KG indicates the possibility of providing not only estimates of the state $x_t$, but also a measure of confidence in this estimate, as the KG can be related to the covariance of the estimate, as initially explored in.

These combined gains of KN over purely MB and DD approaches were recently observed in, which utilized an early version of KN for RT velocity estimation in an autonomous racing car. In such a setup, a NL, MB mixed KF was traditionally used, and suffered from performance degradation due to inherent mismatches in the formulation of the SS model describing the problem. Nonetheless, previously proposed DD techniques relying on RNN for E2E state estimation were not operable in the desired frequencies on the hardware limited vehicle control unit. It was shown in that the application of KN allowed to achieve improved RT velocity tracking compared to MB techniques while being deployed on the control unit of the vehicle.

Our design of KN gives rise to many interesting future extensions. Since we focus here on SS models where the mappings $\gevol$ and $\gobs$ are known up to some approximation errors, a natural extension of KN is to use the data to pre-estimate them, as demonstrated briefly in the numerical study. Another alternative to cope with these approximation errors is to utilize dedicated neural networks to learn these mappings while training the entire model in an E2E fashion. Doing so is expected to allow KN to be utilized in scenarios with analytically intractable SS models, as often arises when tracking based on unstructured observations, e.g., visual observations as in.

While we train KN in a supervised manner using labeled data, the fact that it preserves the operation of the MB EKF that produces a prediction of the next observation $\hat{y}_t|t-1$ for each time instance indicates the possibility of using this intermediate feature for unsupervised training. One can thus envision KN being trained offline in a supervised manner, while tracking variations in the underlying SS model at run-time by online self supervision, following a similar rationale to that used in for deep symbol detection in time-varying communication channels.

Finally, we note that while we focus here on filtering tasks, SS models are used to represent additional related problems such as smoothing and prediction, as discussed in Subsection~. The fact that KN does not explicitly estimate the SS model implies that it cannot simply substitute these parameters into an alternative algorithm capable of carrying out tasks other than filtering. Nonetheless, one can still design DNN-aided algorithms for these tasks operating with partially known SS models as extensions of KN, in the same manner as many MB algorithms build upon the KF. For instance, as the MB KF constitutes the first part of the Rauch-Tung-Striebel smoother, one can extend KN to implement high-performance smoothing in partially known SS models, as we have recently began investigating in. Nonetheless, we leave the exploration of extensions of KN to alternative tasks associated with SS models for future work. 
# Experiments and Results

In this section we present an extensive numerical study of KN\footnote{The source code used in our numerical study along with the complete set of hyperparameters used in each numerical evaluation can be found online at \url{https://github.com/KalmanNet/KalmanNet_TSP}.}, evaluating its performance in multiple setups
and comparing it to various benchmark algorithms:
- In our first experimental study, we consider multiple linear SS models, and compare KN to the MB KF which is known to minimize the MSE in such a setup. We also confirm our design and architectural choices by comparing KN with alternative RNN based E2E state estimators.
- We next consider two NL SS models, a sinusoidal model, and the chaotic LA. We compare KN with the common NL MB benchmarks; namely, the EKF, UKF, and PF.
- In our last study we consider a localization use case based on the Michigan NCLT DS. Here, we compare KN with MB KF that assumes a linear Wiener kinematic model and with a vanilla RNN based E2E state estimator, and demonstrate the ability of KN to track RW dynamics that was not synthetically generated from an underlying SS model.
## Experimental Setting
Throughout the numerical study and unless stated otherwise, in the experiments involving synthetic data, the SS model is generated using diagonal noise covariance matrices; i.e.,

Q=\mathrm{q}^2*I,

R=\mathrm{r}^2*I,

\nu\triangleq\frac{\mathrm{q}^2}{\mathrm{r}^2}.

By, setting $\nu$ to be 0 dB implies that both the state noise and the observation noise have the same variance.
For consistency, we use the term full information for cases where the SS model available to KN and its MB counterparts accurately represents the underlying dynamics. More specifically, KN operates with {full} knowledge of $\gevol$ and $\gobs$, and without access to the noise covariance matrices, while its MB counterparts operate with an accurate knowledge of $Q$ and $R$.
The term partial information refers to the case where KN and its MB counterparts operate with some level of model {mismatch}, where the SS model design parameters do not represent the underlying dynamics accurately (i.e., are not equal to the SS parameters from which the data was generated).
Unless stated otherwise, the metric used to evaluate the performance is the MSE on a $\dB$ scale. In the figures we depict the MSE in $\dB$ versus the inverse observation noise level, i.e., $\frac{1}{\gscal{r} ^2}$, also on a $\dB$ scale. In some of our experiments, we evaluate both the MSE and its standard deviation, where we denote these measures by $\hat{\mu$ and $\hat{\sigma}$, respectively.}

### KN SETTING
In Section~ we present several architectures and training mechanisms that can be used when implementing KN. In our experimental study we consider three different configurations of KN:
- KN ARCH1 with input features \{,\} and with training algorithm.
- KN ARCH1 with input features \{,\} and with training algorithm.
- KN ARCH1 with input features \{,,\} and with training algorithm.
- KN ARCH2 with all input features and with training algorithm.
In all our experiments KN was trained using the Adam optimizer.

### Model-Based Filters
In the following experimental study we compare KN with several MB filters. For the UKF we used the software package, while the PF is implemented based on using 100 particles and without parallelization. During our numerical study, when model uncertainty was introduced, we optimized the performance of the MB algorithms by carefully tuning the covariance matrices, usually via a grid search. For long trajectories (e.g., $T>1500$) it was sometimes necessary to tune these matrices, even in the case of full information, to compensate for inaccurate uncertainty propagation due to NL approximations and to avoid divergence.
## Linear State Space Model
Our first experimental study compares KN to the MB KF for different forms of synthetically generated {linear} system dynamics. Unless stated otherwise, here $F$ takes the
controllable canonical form.

### Full Information
We start by comparing KN of setting to the MB KF for the case of full information, where the latter is known to minimize the MSE. Here, we set $H$ to take the inverse canonical form, and $\nu=0\dB$. To demonstrate the applicability of KN to various linear systems, we experimented with systems of different dimensions; namely, $m* n\in\set{2\times2, 5\times5, 10 * 1}$, and with trajectories of different lengths; namely, $T\in \set{50, 100, 150, 200}$. In Fig.~ we can clearly observe that KN achieves the MMSE of the MB KF. Moreover, to further evaluate the gains of the hybrid architecture of KN, we check that its learning is transferable. Namely, in some of the experiments, we test KN on longer trajectories then those it was trained on, and with different initial conditions. The fact that KN achieves the MMSE lower bound also for these cases indicates that it indeed learns to implement KFING, and it is not tailored to the trajectories presented during training, with dependency only on the SS model.
{Plots_v1/Linear/pdf/sim_Linear_1.pdf}
{KN converges to MMSE.}

{Plots_v1/Linear/pdf/sim_Linear_compare.pdf}
{Learning curves for DD SE.}

{Linear SS model with full information.}
\figSpace

### Neural Model Selection
Next, we evaluate and confirm our design and architectural choices by considering a $2* 2$ setup, and by comparing KN with setting to two RNN based architectures of similar capacity applied for end-to-end state estimation:
- {\em Vanilla RNN} directly maps the observed $y_t$ to an estimate of the state $\hat{x}_t$.
- {\em MB RNN} imitates the Kalman filtering operation by first recovering $\hat{x}_{t\given{t-1}}$ using domain knowledge, i.e., via, and then uses the RNN to estimate an increment $\Delta\hat{x}_t$ from the prior to posterior.
All RNN utilize the same architecture as in KN with a single GRU layer and the same learning hyperparameters. In this experiment we test the trained models on trajectories with the same length as they were trained on, namely $T=20$. We can clearly observe how each of the key design considerations of KN affect the learning curves depicted in Fig.~:
- The incorporation of the known SS model allows the MB RNN to outperform the vanilla RNN, although both converge slowly and fail to achieve the MMSE.
- Using the sequences of differences as input notably improves the convergence rate of the MB RNN, indicating the benefits of using the differences as features, as discussed in Subsection~.
- Learning is further improved by using the RNN for recovering the KG as part of the KF flow, as done by KN, rather than for directly estimating $x_t$.

To further evaluate the gains of KN over end-to-end RNN, we compare the pre-trained models using trajectories with different initial conditions and a longer time horizon ($T=200$) than the one on which they were trained ($T=20$). The results, summarized in Table~, show that KN maintains achieving the MMSE, as already observed in Fig.~. The MB RNN and vanilla RNN are more than $50\dB$ from the MMSE, implying that their learning is not transferable and that they do not learn to implement KFING. However, when provided with the difference features as we proposed in Subsection~, the DD systems are shown to be applicable in longer trajectories, with KN achieving MSE within a minor gap of that achieved by the MB KF. The results of this study validate the considerations used in designing KN for the DD filtering problem discussed in Subsection~.
{Test MSE in [dB] when trained using $T=20$.}

{
\hline
Test T Vanilla RNN MB RNN MB RNN, diff. KN KF 
\hline
20 -20.98 -21.53 -21.92 -21.92 {\bf -21.97}
\hline
200 58.14 36.8 -21.88 -21.90 {\bf -21.91}
\hline

}
\figSpace
### Partial Information
To conclude our study on linear models, we next evaluate the robustness of KN to model mismatch as a result of partial model information. We simulate a $2* 2$ SS model with mismatches in either the state-evolution model ($F$) or in the state-observation model ($H$).

State-Evolution Mismatch: Here, we set $T=20$ and $\nu=0\dB$ and use a rotated evolution matrix $F_\alpha^\circ, \alpha \in \set{10^\circ, 20^\circ}$ for data generation. The state-evolution matrix available to the filters, denoted $F_0$, is again set to take the controllable canonical form. The mismatched design matrix $F_0$ is related to true $F_\alpha^\circ$ via
F_\alpha^\circ=R^{\gscal{xy}}_\alpha^\circ*F_0,

R^{\gscal{xy}}_\alpha^\circ=
\cos{\alpha} -\sin{\alpha} 
\sin{\alpha} \cos{\alpha}
Such scenarios represent a setup in which the analytical approximation of the SS model differs from the true generative model. The resulting MSE curves depicted in Fig.~ demonstrate that KN (with setting) achieves a $3\dB$ gain over the MB KF. In particular, despite the fact that KN implements the KF with an inaccurate state-evolution model, it learns to apply an alternative KG, resulting in MSE within a minor gap from the MMSE; i.e., from the KF with the true $F_\alpha^\circ$ plugged in.

\textcolor{NewColor{State-Observation Mismatch}:} Next, we simulate a setup with state-observation mismatch while setting $T=100$ and $\nu=-20\dB$. The model mismatch is achieved by using a rotated observation matrix $H_\alpha=10^\circ$ for data generation, while using $H=I$ as the observation design matrix. Such scenarios represent a setup in which a slight misalignment $(\approx5\%)$ of the sensors exists. The resulting achieved MSE depicted in Fig.~ demonstrates that KN (with setting) converges to within a minor gap from the MMSE. Here, we performed an additional experiment, first estimating the observation matrix from data, and then KN used the estimate matrix denoted $\hat{H}_\alpha$. In this case it is observed in Fig.~ that KN achieves the MMSE lower bound.
These results imply that KN converges also in distribution to the KF.
{Plots_v1/Linear/pdf/sim_Linear_F_rot.pdf}
{State-evolution mismatch.}

{Plots_v1/Linear/pdf/sim_Linear_H_rot.pdf}
{State-observation mismatch.}

{Linear SS model, partial information.}
\figSpace

## Synthetic Non-Linear Model

Next, we consider a NL SS model, where the state-evolution model takes a sinusoidal form, while the state-observation model is a second order polynomial. The resulting SS model is given by

f\brackets{x} =
\alpha*\sin\brackets{\beta*x+\phi}+\delta,

x\in\greal^2,

h\brackets{x} =
a*\brackets{b*x+c}^2,

y\in\greal^2.
In the following we generate trajectories of $T=100$ time steps from the noisy SS model in, with $\nu=-20\dB$, while using $\gevol$ and $\gobs$ as in computed in a component-wise manner, with parameters as in Table~. KN is used with setting.

The MSE values for different levels of observation noise achieved by KN compared \textcolor{NewColor{with} the MB EKF are depicted in Fig.~ for both full and partial model information. The full evaluation with the MB EKF, UKF, and PF is given in Table~ for the case of full information, and in Table~ for the case of partial information. We first observe that the EKF achieves the lowest MSE values among the MB filters, therefore serving as our main MB benchmark in our experimental studies. For full information and in the low noise regime, EKF achieves the lowest MSE values due to its ability to approach the MMSE in such setups, and KN achieves similar performance. For higher noise levels; i.e., for $\gsnr = -12.04$ [dB], the MB EKF suffers from degraded performance due to a NL effect. Nonetheless, by learning to compute the KG from data, KN manages to overcome this and achieves superior MSE. }

In the presence of partial model information, the state-evolution parameters used by the filters differs slightly from the true model, resulting in a notable degradation in the performance of the MB filters due to the model mismatch. In all experiments, KN overcomes such mismatches, and its performance is within a small gap of that achieved when using full information for such setups. We thus conclude that in the presence of harsh non-linearities as well as model uncertainty due to inaccurate approximation of the underlying dynamics, where MB variations of the KF fail, KN learns to approach the MMSE while maintaining the RT operation and low complexity of the KF.

{{Plots_v1/NL_sin/NL_sin.pdf}}
{Non-linear SS model.
KN outperforms EKF.}

\figSpace
{Non-linear toy problem parameters.}

{
\hline
 $\alpha$ $\beta$ $\phi$ $\delta$ a b c 
\hline
{Full} 0.9 1.1 $0.1\pi$ 0.01 1 1 0 
\hline
{Partial} 1 1 0 0 1 1 0 
\hline

}
\figSpace

{MSE $\dB$ – Synthetic NL SS model; full information.}

{
\hline
\multicolumn{2}{|c|}{ ${1}/{\gscal{r}^2} \dB $} -12.04 -6.02 0 20 40 
\hline
EKF $\hat{\mu}$ -6.23 -13.41 -19.58 -39.78 -59.67

 $\hat{\sigma}$ $\pm 0.89$ $\pm 0.53$ $\pm 0.47$ $\pm 0.43$ $\pm 0.44$

\hline
UKF $\hat{\mu}$ -6.48 -13.14 -18.43 -27.24 -37.27

 $\hat{\sigma}$ $\pm 0.69$ $\pm 0.49$ $\pm 0.50$ $\pm 0.55$ $\pm 0.31$

\hline
PF $\hat{\mu}$ -6.59 -13.33 -18.78	 -26.70 -30.98

 $\hat{\sigma}$ $\pm 0.74$ $\pm 0.48$ $\pm 0.39$ $\pm 0.07$ $\pm 0.02$

\hline
KN $\hat{\mu}$ -7.25 -13.19 -19.22 -39.13 -59.10

 $\hat{\sigma}$ $\pm 0.49$ $\pm 0.52$ $\pm 0.55$ $\pm 0.49$ $\pm 0.53$

\hline

}

{MSE $\dB$ – Synthetic NL SS model; partial information.}

{
\hline
\multicolumn{2}{|c|}{ ${1}/{\gscal{r}^2} \dB $} -12.04 -6.02 0 20 40 
\hline
EKF $\hat{\mu}$ -2.99 -5.07 -7.57 -22.67 -36.55

 $\hat{\sigma}$ $\pm 0.63$ $\pm 0.89$ $\pm 0.45$ $\pm 0.42$ $\pm 0.3$

\hline
UKF $\hat{\mu}$ -0.91 -1.54 -5.18 -24.06 -37.96

 $\hat{\sigma}$ $\pm 0.60$ $\pm 0.23$ $\pm 0.29$ $\pm 0.43$ $\pm 2.21$

\hline
PF $\hat{\mu}$ -2.32 -3.29 -4.83 -23.66 -33.13

 $\hat{\sigma}$ $\pm 0.89$ $\pm 0.53$ $\pm 0.64$ $\pm 0.48$ $\pm 0.45$

\hline
KN $\hat{\mu}$ -6.62 -11.60 -15.83 -34.23 -45.29

 $\hat{\sigma}$ $\pm 0.46$ $\pm 0.45$ $\pm 0.44$ $\pm 0.58$ $\pm 0.64$

\hline

}

## Lorenz Attractor
The LA is a three-dimensional chaotic solution to the Lorenz system of ordinary differential equations in CT. This synthetically generated system demonstrates the task of online tracking a highly NL trajectory and a RW practical challenge of handling mismatches due to sampling a {CT} signal into DT~.

In particular, the noiseless state-evolution of the CT process $x_\tau$ with $\tau\in\greal^+$ is given by
(\partial)/(\partial \tau){x}_\tau\!=\!
A\brackets{{x}_\tau}* x_\tau,

A\brackets{{x_\tau}}\!=\!
-10 10 0
28 -1 -x_1, \tau
0 x_1, \tau -(8)/(3)
To get a DT, state-evolution model, we repeat the steps used in. First, we sample the noiseless process with sampling interval $\Delta\tau$
and assume that $A\brackets{{x}_\tau}$ can be kept constant in a small neighborhood of $x_\tau$; i.e.,
A\brackets{{x}_\tau}\approx A\brackets{{x}_\tau+\Delta\tau}.
Then, the CT solution of the differential system, which is valid in the neighborhood of $x_\tau$ for a short time interval $\Delta\tau$, is
x_\tau+\Delta\tau=
\exp\brackets{A\brackets{{x_\tau}}*\Delta\tau}*
x_\tau.
Finally, we take the Taylor series expansion of and a finite series approximation, which results in
F\brackets{{x_\tau}}\triangleq
\exp\brackets{A\brackets{{x_\tau}}*\Delta \tau}\approx
I + \sum_j=1^J \frac{\brackets{A\brackets{{x_\tau}}*\Delta \tau}^j}{j!}.
The resulting DT evolution process is given by
x_t+1 = f\brackets{x_t}=
F\brackets{{x_t}}*x_t.
The DT state-evolution model in, with additional process noise, is used for generating the simulated LA data. Unless stated otherwise the data was generated with $J=5$ Taylor order and $\Delta \tau = 0.02$ sampling interval. In the following experiments, KN is consistently invariant of the distribution of the noise signals, with the models it uses for $\gevol$ and $\gobs$ varying between the different studies, as discussed in the sequel.
\medskip

### Full Information
We first compare KN to the MB filter when using the state-evolution matrix $F$ computed via with $J=5$.
\smallskip

{\bf Noisy state observations}: Here, we set $\gobs$ to be the identity transformation, such that the observations are noisy versions of the true state. Further, we set $\nu=-20\dB$ and $T=2000$. As observed in Fig.~, despite being trained on short trajectories $T=100$, KN (with setting) achieves excellent MSE performance—namely, comparable to EKF—and outperforms the UKF and PF. The full details of the experiment are given in Table~. All the MB algorithms were optimized for performance; e.g., applying the EKF with full model information achieves an unstable state tracking performance, with MSE values surpassing $30\dB$. To stabilize the EKF, we had to perform a grid search using the available data set to optimize the process noise $Q$ used by the filter.

{MSE $\dB$ – LA with noisy state observations.}

{
\hline
${1}/{\gscal{r}^2} \dB$ 0 10 20 30 40 
\hline
EKF -10.45 -20.37 -30.40 -40.39 -49.89 
\hline
UKF -5.62 -12.04 -20.45 -30.05 -40.00 
\hline
PF -9.78 -18.13 -23.54 -30.16 -33.95 
\hline
KN -9.79 -19.75 -29.37 -39.68 -48.99 
\hline

}
\smallskip

{\bf {Noisy NL observations}}: Next, we consider the case where the observations are given by a non-linear function of the current state, setting $h$ to take the form of a transformation from a cartesian coordinate system to spherical coordinates. We further set $T=20$ and $\nu=0\dB$. From the results depicted in Fig.~ and reported in Table~ we observe that in such non-linear setups, the sub-optimal MB approaches operating with full information of the SS model are substantially outperformed by KN (with setting).

{MSE $\dB$ – LA with non-linear observations}

{
\hline
${1}/{\gscal{r}^2} \dB $ -10 0 10 20 30
\hline
EKF 26.38 21.78 14.50 4.84 -4.02
\hline
UKF \textrm{nan}	 \textrm{nan} \textrm{nan} \textrm{nan} \textrm{nan} 
\hline
PF 24.85 20.91 14.23 11.93 4.35 
\hline
KN 14.55 6.77 -1.77 -10.57 -15.24 
\hline

}
{Plots_v1/Lorenz/pdf/sim_Lorenz_H_I.pdf}
{$T=2000$, $\nu=-20\dB$, $\gobs=I$.}

{Plots_v1/Lorenz/pdf/sim_Lorenz_H_NL.pdf}
{$T=20$, $\nu=0\dB$, $\gobs$ NL.}

{LA, full information.}

\figSpace

\smallskip

### Partial Information
W proceed to evaluate KN and compare it to its MB counterparts under partial model information. We consider three possible sources of model mismatch arising in the LA setup:
- State-evolution mismatch due to use of a Taylor series approximation of insufficient order.
- State-observation mismatch as a result of misalignment due to rotation.
- State-observation mismatch as a result of sampling from CT to DT.

Since the EKF produced the best results in the full information case among all NL MB filtering algorithms, we use it as a baseline for the MSE lower bound.
\smallskip

{\bf State-evolution mismatch}:
In this study, both KN and the MB algorithms operate with a crude approximation of the evolution dynamics obtained by computing with $J=2$, while the data is generated with an order $J=5$ Taylor series expansion. We again set $h$ to be the identity mapping, $T=2000$, and $\nu=-20\dB$. The results, depicted in Fig.~ and reported in Table, demonstrate that KN (with setting) learns to partially overcome this model mismatch, outperforming its MB counterparts operating with the same level of partial information.

{MSE $\dB$ - LA with state-evolution mismatch $J=2$.}

{
\hline
\multicolumn{2}{|c|}{ ${1}/{\gscal{r}^2} \dB $} 10 20 30 40
\hline
EKF $\hat{\mu}$ -20.37 -30.40 -40.39 -49.89
 $J=5$ $\hat{\sigma}$ $\pm 0.25$ $\pm 0.24$ $\pm 0.24$ $\pm 0.20$
\hline
EKF $\hat{\mu}$ -19.47 -23.63 -33.51 -41.15
 $J=2$ $\hat{\sigma}$ $\pm 0.25$ $\pm 0.11$ $\pm 0.18$ $\pm 0.12$
\hline
UKF $\hat{\mu}$ -11.95 -20.45 -30.05 -39.98
 $J=2$ $\hat{\sigma}$ $\pm 0.87$ $\pm 0.27$ $\pm 0.09$ $\pm 0.09$
\hline
PF $\hat{\mu}$ -17.95 -23.47	 -30.11 -33.81 
 $J=2$ $\hat{\sigma}$ $\pm 0.18$ $\pm 0.09$ $\pm 0.10$ $\pm 0.13$
\hline
KN $\hat{\mu}$ -19.71 -27.07 -35.41 -41.74 
 $J=2$ $\hat{\sigma}$ $\pm 0.29$ $\pm 0.18$ $\pm 0.20$ $\pm 0.11$
\hline

}
\smallskip

{\bf State-observation rotation mismatch}:
Here, the presence of mismatch in the observations model is simulated by
using data generated by an identity matrix rotated by merely $\theta=1^\circ$. This rotation is equivalent to sensor misalignment of $\approx0.55\%$. The results depicted in Figure.~ and reported in Table~ clearly demonstrate that this seemingly minor rotation can cause a severe performance degradation for the MB filters, while KN (with setting) is able to learn from data to overcome such mismatches and to notably outperform its MB counterparts, which are sensitive to model uncertainty. Here, we trained KN on short trajectories with $T=100$ time steps, tested it on longer trajectories with $T=1000$ time steps, and set $\nu=-20\dB$. This again demonstrates that the learning of KN is transferable.

{MSE $\dB$ - LA with observation rotation.}

{
\hline
\multicolumn{2}{|c|}{ ${1}/{\gscal{r}^2} \dB $} 0 10 20 30
\hline
EKF $\hat{\mu}$ -10.40 -20.41 -30.50 -40.45 
 $\theta=0^\circ$ $\hat{\sigma}$ $\pm 0.35$ $\pm 0.37$ $\pm 0.34$ $\pm 0.34$
\hline
EKF $\hat{\mu}$ -9.80 -16.50 -18.19	 -18.57 
 $\theta=1^\circ$ $\hat{\sigma}$ $\pm 0.54$ $\pm 6.51$ $\pm 0.22$ $\pm 0.21$
\hline
UKF $\hat{\mu}$ -2.08 -6.92 -7.89 -8.09 
 $\theta=1^\circ$ $\hat{\sigma}$ $\pm 1.73$ $\pm 0.53$ $\pm 0.59$ $\pm 0.62$
\hline
PF $\hat{\mu}$ -8.48 -0.18 15.24 19.87 
 $\theta=1^\circ$ $\hat{\sigma}$ $\pm 3$ $\pm 8.21$ $\pm 3.50$ $\pm 0.80$
\hline
KN $\hat{\mu}$ -9.63 -18.17 -27.32 -34.04
 $\theta=1^\circ$ $\hat{\sigma}$ $\pm 0.53$ $\pm 0.42$ $\pm 0.67$ $\pm 0.77$
\hline

}
{Plots_v1/Lorenz/pdf/sim_Lorenz_J_2.pdf}
{State-evolution mismatch, identity $h$, $T=2000$.}

{Plots_v1/Lorenz/pdf/sim_Lorenz_H_rot.pdf}
{Observation mismatch - $\Delta\theta=1^\circ$, $T=1000$.}

{LA, partial information.}
\figSpace

\smallskip

{\bf State-observations sampling mismatch}:
We conclude our experimental study of the LA setup with an evaluation of KN in the presence of sampling mismatch. Here, we generate data from the LA SS model with an approximate CT evolution process using a dense sampling rate, set to ${\Delta\tau=10^-5}$. We then sub-sample the noiseless observations from the evolution process by a ratio of $(1)/(2000)$ and get a decimated process with $\Delta\tau_{\textrm{d}}=0.02$. This procedure results in an inherent mismatch in the SS model due to representing an CT process using a DT sequence. In this experiment, no process noise was applied, and the observations are again obtained with $h$ set to identity and $T=3000$.

The resulting MSE values for $\frac{1}{\gscal{r}^2}=0\dB$ of KN with configuration compared with the MB filters and with the end-to-end neural network termed MB-RNN (see Subsection~) are reported in Table~. The results demonstrate that KN overcomes the mismatch induced by representing a CT SS model in DT, achieving a substantial processing gain over the MB alternatives due to its learning capabilities. The results also demonstrate that KN significantly outperforms a straightforward combination of domain knowledge; i.e. a state-transition function $\gevol$, with end-to-end RNN. A fully model-agnostic RNN was shown to diverge when trained for this task.
In Fig.~ we visualize how this gain is translated into clearly improved tracking of a single trajectory. To show that these gains of KN do not come at the cost of computationally slow inference, we detail the average inference time for all filters. The stopwatch timings were measured on the same platform -- Google Colab with CPU: Intel Xeon CPU @ 2.20GHz, GPU: Tesla P100-PCIE-16GB. We see that KN infers faster than the classical methods, thanks to the highly efficient neural network computations and the fact that, unlike the MB filters, it does not involve linearization and matrix inversions for each time step.

{LA with sampling mismatch.}
{
\hline
$\textrm{Metric}$ EKF UKF PF KN MB-RNN
\hline
MSE $\dB$ -6.432 -5.683 -5.337 {\bf -11.284} 17.355
 $\hat{\sigma}$ $\pm 0.093$ $\pm 0.166$ $\pm 0.190$ $\pm 0.301$ $\pm 0.527$
 \hline
 \textrm{Run-time} $[\sec]$ 5.440
 6.072 62.946
 {\bf 4.699} 2.291
 \hline

}
\figSpace
## Real World Dynamics: Michigan NCLT Data Set
In our final experiment we evaluate KN on the Michigan NCLT DS. This DS comprises different labeled trajectories, with each one containing noisy sensor readings and the ground truth locations of a moving Segway robot. Given these noisy readings, the goal of the tracking algorithm is to localize the Segway from the raw measurements at any given time.

To tackle this problem we model the Segway kinematics using the linear Wiener velocity model, where the acceleration is modeled as a white Gaussian noise process $w_\tau$ with variance $\gscal{q^2$}:
x_\tau=\brackets{p, v}^\top\in\greal^2,

(\partial)/(\partial \tau) {x}_\tau=
0 1 
0 0
0 
w_\tau
Here, p and v are the position and velocity, respectively. The DT state-evolution with sampling interval $\Delta \tau$ is approximated as a linear SS model in which the evolution matrix $F$ and noise covariance $Q$ are given by
F =
1 \Delta \tau
0 1

Q = \gscal{q^2}*
(1)/(3)*\brackets{\Delta \tau}^3 (1)/(2)*\brackets{\Delta \tau}^2 
(1)/(2)*\brackets{\Delta \tau}^2 \Delta \tau
Since KN does not rely on knowledge of the noise covariance matrices, $Q$ is given here for the use of the MB KF and for completeness.

The goal is to track the underlying state vector in both axes solely using odometry data; i.e., the observations are given by noisy velocity readings. In this case the observations obey a noisy linear model:
\gscal{y}\in\greal,

H =\brackets{0, 1}.
Such settings where one does not have access to direct measurements for positioning are very challenging yet practical and typical for many applications where positioning technologies are not available indoors, and one must rely on noisy odometer readings for self-localization. Odometry-based estimated positions typically start drifting away at some point.

In the assumed model, the x-axis are decoupled from the y-axis, and the linear SS model used for Kalman filtering is given by
\tilde{F} =
F 0
0 F

\tilde{Q} =
Q 0
0 Q
\tilde{H} =
H 0
0 H

\tilde{R} =
\gscal{r}^2 0
0 \gscal{r^2}
This model is equivalent to applying two independent KF in parallel. Unlike the MB KF, KN does not rely on noise modeling, and can thus accommodate dependency in its learned KG.

We arbitrarily use the session with date 2012-01-22 that consists of a single trajectory. Sampling at $1\textrm{[Hz]}$ results in 5,850 time steps. We removed unstable readings and were left with 5,556 time steps. The trajectory was split into three sections: $85\%$ for training (23 sequences of length $T=200$), $10\%$ for validation (2 sequences, $T=200$), and $5\%$ for testing (1 sequence, $T=277$). We compare KN with setting to end-to-end vanilla RNN and the MB KF, where for the latter the matrices {$Q$} and $R$ were optimized through a grid search.

Fig. and Table~ demonstrate the superiority of KN for such scenarios. KF blindly follows the odometer trajectory and is incapable of accounting for the drift, producing a very similar or even worse estimation than the integrated velocity. The vanilla RNN, which is agnostic of the motion model, fails to localize. KN overcomes the errors induced by the noisy odometer observations, and provides the most accurate RT locations, demonstrating the gains of combining MB KF-based inference with integrated DD modules for RW applications.
{Numerical MSE $\dB$ for the NCLT experiment.}
{
 \hline
 Baseline EKF KN Vanilla RNN
 \hline
25.47 25.385 {\bf 22.2} 40.21 
 \hline

}
\figSpace
{Plots_v1/Decimation/Decimation_11.pdf}
{Plots_v1/Decimation/Decimation_21.pdf}
{LA with sampling mismatch, $T=3000$.}

\figSpace

{Plots_v1/NCLT/NCLT_traj.png}
{NCLT DS: ground truth vs. integrated velocity, trajectory from session with date 2012-01-22 sampled at 1 Hz.}

# Conclusions
In this work we presented KN, a hybrid combination of DL with the classic MB EKF. Our design identifies the SS-model-dependent computations of the MB EKF, replacing them with a dedicated RNN operating on specific features encapsulating the information needed for its operation.
Our numerical study shows that doing so enables KN to carry out RT SE in the same manner as MB Kalman filtering, while learning to overcome model mismatches and non-linearities.
KN uses a relatively compact RNN that can be trained with a relatively small DS and infers a reduced complexity, making it applicable for high dimensional SS models and computationally limited devices.
# Acknowledgements
We would like to thank Prof. Hans-Andrea Loeliger for his helpful comments and discussions, and Jonas E. Mehr for his assistance with the numerical study.
\bibliographystyle{IEEEtran}
\bibliography{IEEEabrv,KalmanNet}