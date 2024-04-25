---
title: "Bridging the Data Processing Inequality and Function-Space Variational Inference"
author: Andreas Kirsch, University of Oxford\textsuperscript{--2023}
documentclass: tikzposter
classoption:
  - 25pt
  - a0paper
  - landscape
header-includes:
  - \usepackage{tcolorbox}
  - \tcbuselibrary{skins}
  - \usepackage{graphicx}
  - \usepackage{multicol}
  - \usepackage{amsmath,amsfonts,amssymb,amsthm,bm} 
  - \usepackage{mathtools}
  - |
    \usepackage{xparse}
    \newcommand{\hideFromPandoc}[1]{#1}
        \hideFromPandoc{
            \let\Begin\begin
            \let\End\end
            % \NewDocumentCommand{\BeginTColorBox}{o}{\begin{tcolorbox}[#1]}
            % \NewDocumentCommand{\EndTColorBox}{}{\end{tcolorbox}}
        }
  - |
    \DeclareMathOperator{\opExpectation}{\mathbb{E}}
    \newcommand{\E}[2]{\opExpectation_{#1} \left [ #2 \right ]}
    \newcommand{\simpleE}[1]{\opExpectation_{#1}}
    \newcommand{\MidSymbol}[1][]{\:#1\:}
    \newcommand{\given}{\MidSymbol[\vert]}
    \DeclareMathOperator{\opmus}{\mu^*}
    \newcommand{\IMof}[1]{\opmus[#1]}
    \DeclareMathOperator{\opInformationContent}{H}
    \newcommand{\ICof}[1]{\opInformationContent[#1]}
    \newcommand{\xICof}[1]{\opInformationContent(#1)}
    \DeclareMathOperator{\opEntropy}{H}
    \newcommand{\Hof}[1]{\opEntropy[#1]}
    \newcommand{\xHof}[1]{\opEntropy(#1)}
    
    \DeclareMathOperator{\opMI}{I}
    \newcommand{\MIof}[1]{\opMI[#1]}
    \DeclareMathOperator{\opTC}{TC}
    \newcommand{\TCof}[1]{\opTC[#1]}
    \newcommand{\CrossEntropy}[2]{\opEntropy(#1 \MidSymbol[\Vert] #2)}
    \DeclareMathOperator{\opKale}{D_\mathrm{KL}}
    \newcommand{\Kale}[2]{\opKale(#1 \MidSymbol[\Vert] #2)}
    \DeclareMathOperator{\opJSD}{D_\mathrm{JSD}}
    \newcommand{\JSD}[2]{\opJSD(#1 \MidSymbol[\Vert] #2)}
    \newcommand{\opp}{\mathrm{p}}
    \newcommand{\pof}[1]{\opp(#1)}
    \newcommand{\hpof}[1]{\hat{\opp}(#1)}
    \newcommand{\pcof}[2]{\opp_{#1}(#2)}
    \newcommand{\hpcof}[2]{\hat\opp_{#1}(#2)}
    \newcommand{\opq}{\mathrm{q}}
    \newcommand{\qof}[1]{\opq(#1)}
    \newcommand{\hqof}[1]{\hat{\opq}(#1)}
    \newcommand{\qcof}[2]{\opq_{#1}(#2)}
    \newcommand{\varHof}[2]{\opEntropy_{#1}[#2]}
    \newcommand{\xvarHof}[2]{\opEntropy_{#1}(#2)}
    \newcommand{\varMIof}[2]{\opMI_{#1}[#2]}
    \newcommand{\w}{\boldsymbol{\theta}}
    \newcommand{\W}{\boldsymbol{\Theta}}
    \newcommand{\opf}{\mathrm{f}}
    \newcommand{\fof}[1]{\opf(#1)}
    \newcommand{\Dany}{\mathcal{D}}
    \newcommand{\y}{y}
    \newcommand{\Y}{Y}
    \newcommand{\Loss}{\boldsymbol{L}}
    \newcommand{\x}{\boldsymbol{x}}
    \newcommand{\X}{\boldsymbol{X}}
    \newcommand{\pdata}[1]{\hpcof{\text{data}}{#1}}
    \newcommand{\normaldist}[1]{\mathcal{N}(#1)}
---

<!-- To render: pandoc -i _posts/fsvi_poster.md -o fsvi_poster.pdf --from markdown --to pdf -->

<!-- documentclass: beamer
header-includes:
  - \usepackage[orientation=landscape,size=a0,scale=1.0,debug]{beamerposter} -->

\Begin{multicols}{2}

\centering
# Data Processing Inequalities

\Begin{multicols}{2}

\begin{tcolorbox}[colback=green!5!white,colframe=teal!75!black,title=KL TK Data Processing Inequality]
The DPI states that processing data stochastically can only reduce information. Formally, for distributions $\qof{\W}$ and $\pof{\W}$ over a random variable $\W$ and a stochastic mapping $Y = \fof{\W}$, the DPI is expressed as:

$$\Kale{\qof{\W}}{\pof{\W}} \ge \Kale{\qof{Y}}{\pof{Y}}$$

Equality holds when $\Kale{\qof{\W \given Y}}{\pof{\W \given Y}} = 0$.
\end{tcolorbox}

\begin{tcolorbox}[colback=green!5!white,colframe=teal!75!black,title=Proof]
Using the chain rule of the KL divergence:

$$
\begin{aligned}
\Kale{\pof{X, Y}}{\qof{X, Y}} &= \Kale{\pof{X}}{\qof{X}} \\
&+ \Kale{\pof{Y \given X}}{\qof{Y \given X}},
\end{aligned}
$$

and its symmetry, we have:

$$
\begin{aligned}
&\Kale{\pof{X}}{\qof{X}} + \underbrace{\Kale{\pof{Y\given X}}{\qof{Y \given X}}}_{=\Kale{\fof{Y\given X}}{\fof{Y \given X}}=0}\\
&\quad =\Kale{\pof{X, Y}}{\qof{X, Y}}\\
&\quad =\Kale{\pof{Y}}{\qof{Y}}+\underbrace{\Kale{\pof{X \given Y}}{\qof{X \given Y}}}_{\ge 0}\\
&\quad \ge \Kale{\pof{Y}}{\qof{Y}}.
\end{aligned}
$$

We have equality exactly when $\pof{x \given y} = \qof{x \given y}$ for (almost) all $x, y.$
\end{tcolorbox}

\columnbreak

\begin{tcolorbox}[drop fuzzy shadow southeast,enhanced,colback=brown!5!orange!40!white, colframe=brown!80!black, boxrule=1pt, left=5pt, right=5pt, top=5pt, bottom=5pt]
{\Large\emph{The data processing inequality states that if two random variables are transformed in this way, they cannot become easier to tell apart.}}
\begin{flushright}
``Understanding Variational Inference in Function-Space'', \\
Burt et al. (2021)
\end{flushright}
\end{tcolorbox}

\begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title=Example: Image Processing]
Consider an image processing pipeline where $X$ is the original image, $Y$ is a compressed version, and $Z$ is $Y$ after adding blur and pixelation. The DPI tells us that $\MIof{X;Y} \ge \MIof{X;Z}$, as each processing step results in information loss.
\end{tcolorbox}


TK examples TK

\End{multicols}

\begin{tcolorbox}[colback=green!5!white,colframe=teal!75!black,title=From KL to Jenson-Shannon Divergence to Mutual Information]

\begin{multicols}{2}
\subsubsection{Jensen-Shannon Divergence}

The Jensen-Shannon divergence (JSD) makes the KL divergence symmetric:

$$
\begin{aligned}
\fof{x} &= \frac{\pof{x} + \qof{x}}{2}\\
\JSD{\pof{x}}{\qof{x}} &= \frac{1}{2} \Kale{\pof{x}}{\fof{x}} + \frac{1}{2} \Kale{\qof{x}}{\fof{x}}.
\end{aligned}
$$

The square root of the Jensen-Shannon divergence, the \emph{Jensen-Shannon distance}, is symmetric, satisfies the triangle inequality and hence a metric.

Given $\pof{x}$ and $\qof{x}$ and shared transition function $\fof{y \given x}$ for the model $X \rightarrow Y$, we obtain a Jensen-Shannon divergence data processing inequality by applying the KL DPI twice:

$$
\JSD{\pof{X}}{\qof{X}} \ge \JSD{\pof{Y}}{\qof{Y}}.
$$

\subsubsection{Mutual Information}

For 
$$
\begin{aligned}
Z &\sim \mathrm{Bernoulli}(\frac{1}{2}) = \fof{Z} \\
X \given Z = 0 \sim \pof{x} &\land X \given Z = 1 \sim \qof{x},
\end{aligned}
$$
$$
\begin{aligned}
&\MIof{X;Z} = \Kale{\fof{X \given Z}}{\fof{X}} = \E{\fof{z}} {\Kale{\fof{X \given Z = z}}{\fof{X}}}\\
&= \frac{1}{2} \Kale{\pof{x}}{\fof{x}} + \frac{1}{2} \Kale{\qof{x}}{\fof{x}}= \JSD{\pof{X}}{\qof{X}}.
\end{aligned}
$$

We can generalize this to the Markov chain $Z \rightarrow X \rightarrow Y$ with $\fof{z, x, y} = \fof{z} \fof{x \given z} \fof{y \given x}$ for any distribution $\fof{z}$:

$$
\begin{aligned}
\MIof{X;Z} &= \Kale{\fof{X \given Z}}{\fof{X}} = \E{\fof{z}} {\Kale{\fof{X \given z}}{\fof{X}}}\\
&\overset{(1)}{\ge} \E{\fof{z}} {\Kale{\fof{Y \given z}}{\fof{Y}}} = \Kale{\fof{Y \given Z}}{\fof{Y}}\\
&= \MIof{Y;Z},
\end{aligned}
$$

where $(1)$ follows from the KL data processing inequality. 
%This is the data processing inequality for the mutual information.
\end{multicols}
\end{tcolorbox}

\columnbreak

\centering
# Function-Space Variational Inference

FSVI is a principled approach to Bayesian inference that respects the inherent symmetries and equivalences in overparameterized models. It approximates the posterior over equivalence classes of parameters $\pof{[\w] \given \Dany}$ using an implicit variational distribution $\qof{[\w]}$. FSVI minimizes $\Kale{\qof{[\W]}}{\pof{[\W] \given \Dany}}$, which is invariant to parameter symmetries.

The FSVI-ELBO regularizes towards a data prior:
$$\E{\qof{\w}}{-\log \pof{\Dany \given \w}} + \Kale{\qof{\Y... \given \x...}}{\pof{\Y... \given \x...}}$$

\begin{tcolorbox}[colback=green!5!white,colframe=green!75!black,title=FSVI and Parameter Symmetries]
FSVI sidesteps the need to explicitly define equivalence classes or specify a model that operates on them directly. By using an implicit variational distribution and leveraging the DPI, FSVI can approximate the meaningful posterior $\pof{[\w] \given \Dany}$ while avoiding the complexities of working with equivalence classes.
\end{tcolorbox}

# Connecting DPI and FSVI

The connection between DPI and FSVI allows FSVI to measure a predictive divergence independent of parameter symmetries. By matching predictive priors in the limit of infinite data, FSVI effectively matches posteriors over equivalence classes. This insight relates FSVI to label entropy regularization and knowledge distillation.

\begin{tcolorbox}[colback=red!5!white,colframe=red!75!black,title=Relation to Other Methods]
The connection between DPI and FSVI highlights the practical relevance of these theoretical concepts. It shows how FSVI relates to training with knowledge distillation and label entropy regularization, providing a new perspective on these methods.
\end{tcolorbox}

\begin{multicols}{2}

Conclusion

The data processing inequality provides a powerful intuition about the limitations of information processing systems. Function-space variational inference respects inherent model symmetries by focusing on the predictive posterior. Examining the connection between DPI and FSVI offers valuable insights for both theory and practice in Bayesian deep learning.

Understanding this connection can guide the development of more principled and effective inference techniques. It highlights the importance of considering the function space and its equivalences, rather than solely focusing on the parameter space.

Future research could further explore the implications of this connection for topics such as active learning, model compression, and transfer learning. By leveraging the insights from DPI and FSVI, we can develop more robust and efficient machine learning methods.

\end{multicols}

\End{multicols}

# References

- Kirsch, A. (2023). Bridging the Data Processing Inequality and Function-Space Variational Inference. \textit{Blog post}. Retrieved from \url{https://www.blackhc.net/blog/2023/data-processing-inequality-and-fsvi/}
- Kirsch, A., & Gal, Y. (2022). Function-Space Variational Inference. \textit{arXiv preprint arXiv:2206.09988}.
- Cover, T. M., & Thomas, J. A. (2006). \textit{Elements of Information Theory}. John Wiley & Sons.