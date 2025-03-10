# ::RAG-Survey::

> The issues that need attention in these notes:

1. > The achievements RAG has already made.
2. > #### The mainstream methods and the representative papers.
> #### Q & A

> The key  to success in recent Q & A is the  introduction of RAG.

> The landmark work is the RAG.

> Retrieval - augmented Generation for Knowledge -Intensive NLP tasks (Patrick Lewis)

> [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

   > This is a work prior to the GPT. Its main target is to refine the result of pretrained models.

      > input: query (q)

      > latent state: retrieved documents (z)

      > output: respond(y)

   > 2 parts:

      1. > The retriever.
         - > retrieve top k w.r.t. cos distance in vector space.
         - > Retriever structure. Dense Passage Retrieval(DPR)
            - > embedding-based.
            - > dual encoder. two berts. one process the query and the other process the documents.
            - > Training.
               - > query encoder : x ⇒ z'
               - > document encoder d ⇒ z
               - > match z' and z as much as possible
$$
p_{\eta}(z|x) \propto \exp(d(z)^{T}q(x))\\
d(z) = BERT_{d}(z)\\
q(x) = BERT_{q}(x)
$$

      1. > generater. BART
         - > input from the retievers output(mixture of laten representation)

   > Training.(or fintuning)

      > jointly training the retriver and generator.

      > only train the generator and the qurey encoder. fix the document encoder.

> Another is FiD .  For open domain question and answering.

> #### Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering(FiD) [https://arxiv.org/abs/2007.01282](https://arxiv.org/abs/2007.01282)

   > more recieve passage?

![Image.png](https://res.craft.do/user/full/8282f642-116f-930d-eb75-ef43f16d982d/doc/D8EA4993-B7AB-4A43-A14C-A4FE40D61BE9/DED80B9E-F8CA-4021-A1A5-CE2D47DBBDDE_2/HdGDEHCoX2qOm7bdy5PxUj19RyKPYd07IYbYAgu35Mkz/Image.png)

> The structure are simliar with RAG. There are some difference:

   - > the question and the passage retrieved as cancated
   - > each pair are fw into a encoder independently
   - > fusion-in-decoder

> REALM- Retrieval-Augmented Language Model Pre-Training(Kelvin Guu, Mingwei Chang)

[Arxiv](https://arxiv.org/pdf/2002.08909)

   > This work should be the first one training in unsuprvised way.

   - > masked language model.
   - > train a neural retriever.

   > main contribution.

      > computational challenge(filter from millions of documentations).

      - > cache and asyn updating, this is formulated as maximum inner product search(MIPS)

   > Training.

      - > pre-training. masked language modeling.
      - > fine-tuning. Q and A.

   > Structure.

      - > retrieve.  p(z|x)
$$
p(z|x) = \frac{\exp f(x,z)}{\sum _{z'}\exp f(x,z')}\\
f(x,z) = Embed_{query}(x)^{T} Embed_{doc}(z)
$$

      - > Compared with RAG, this work change the embedding representation a little bit.
         - > it use a linear prob to reduce the dimensions.
$$
join_{BERT}(x) = [CLS] x [SEP]\\
join_{BERT}(x_1, x_2) = [CLS] x_1 [SEP] x_2[SEP]\\
Embed_{input}(x) = \textcolor{red}{W_{input}} BERT_{CLS}(join_{BERT}(x))\\
Embed_{doc}(x) = \textcolor{red}{W_{doc}} BERT_{CLS}(join_{BERT}(z_{title},z_{body}))
$$

      - > Knowledge augmented encoder.
         - > for masked language model.
$$
p(y_i|z,x) \propto \exp(\omega_{j}^T BERT_{mask(j)}(join_{BERT}(x,z_{body}))\\
p(y|z,x) = \prod_{j=1}^{J_x}p(y_j|z,x)
$$

         - > for Q & A fine-tuning. It produces the answer string y.
            - > y can be found as a contiguous sequence of tokens in some documents.
            - > S(z,y) represents the set of spans matching y in z.
            - > where BERT_START(s) and BERT_END(s) denote the Transformer output vectors corresponding to the start and end tokens of span s
$$
p(y|z,x) \propto \sum_{s\in S(z,y)} \exp (MLP([h_{START(s)}, h_{END(s)}]))\\
h_{START} = BERT_{START(s)}(join_{BERT}(x,z_{body}))\\
h_{END} = BERT_{END(s)}(join_{BERT}(x,z_{body}))
$$

      - > predict.  p(y|z,x) * p(z|x) then marginalize z
$$
p(y|x) = \int_{z\in \mathcal{Z}}p(y|z,x)p(z|x)dz
$$

         - > Only challenging is the computational burden of the marginalization.
         - > This is estimated by the top-k documents.
         - > Even with the top-k estimation, we still need an efficient way to find the top k documents.
         - > since f(x,z) has the form of inner product, it can be classified as a MIPS problem.
            - > calculate document embeddings. Embed_doc(z)
            - > construct an efficient search index over these embeddings.
               - > !! If the doc encoder's parameters are changed during training, the embedding could be stale.
               - > How to solve this. refresh index and embeddings after hundreds training steps.

> Large Dual Encoders Are Generalizable Retrievers(jianmo Ni, Mingwei-chang, yinfei yang, goolge research)

> [https://arxiv.org/pdf/2112.078992112.07899 (arxiv.org)](https://arxiv.org/pdf/2112.07899)

   - > It inherit the same dual encoder structure.
   - > debate against the claim that the embedding dimension is the bottle neck.
   - > scale the T5 up to 11B but keep the same embedding dimension.

> #### Training.

   > Stage one. pre-training on the web-mined corpus.

      > Q & A and conversation.

   > Stage two. Fine-tuning.

      > high quality search corpus.

         > MS Macro

         > Natural Questions

3. > The main issues currently faced.
   1. > hallucination or Irrelevance
> #### Self-RAG: Learning To Retrieve Generate and Critique Through Self-Reflection.

   > by Akari Asai,... Yizhong wang, Hannaneh Hajishirzi.

   > [https://arxiv.org/pdf/2310.115112310.11511 (arxiv.org)](https://arxiv.org/pdf/2310.11511)

   > #### main idea.

      > The process of retrieval should be flexible.

      > let the model itself to decide to retrieve or not

         > my understanding. This is a flexible factor which maybe related to the task.

         - > For more creative task, there could be more flexible of the generation process.
            - > Just let the generator to create some context without any retrieval.
         - > For the knowledge-intensive task, it may help to reduce the factual error when retrieving some relevant information. However, the standard retrieved passage could be irrelevant.
            - > The Generator generate some reflection tokens. These tokens can be used to rank the candidates. Then only the relevant and supporting ones will be selected.
            - > They claim that they can get better performance.

   > #### main components.

   - > Retriever. Pre-Trained model. ⇒ Contriver-MS MARCO
   - > Generator.
      - > auxiliary LM (**critic**). To help to generate reflection tokens as guideness
         - > as ranking metric
         - > as instruction. (e.g whether retrieve a new token or not)

   > #### Training.

      > Retriever. No training needed.

      > Critic.

         > Critic's task is to generate a series of reflection tokens to guide the subsequent behavior of the model. The specific training methods involves from GPT. The reflection tokens generated by GPT is used as a soft label to form the training data. Then the training is carried out using the normal language model loss.{(r, x, y)}

      > Generator.

         > The generator is trained by the **NEW reflection tokens generated by the fully trained Critic.**

   > #### Inference.

      > input ⇒[Retrive?] ⇒ yes? ⇒ Retrieve :⇒ d + ...+  [is_Rel]+y

               > 🔥⇒ /[is_sup]+ [is_use] ⇒ rank y w.r.t. [tokens]

   > ⇒ No! ⇒  Generate :⇒ 🦀 y + [is_use]

![Image.png](https://res.craft.do/user/full/8282f642-116f-930d-eb75-ef43f16d982d/doc/D8EA4993-B7AB-4A43-A14C-A4FE40D61BE9/C35F3E55-CBEE-4141-87FD-DB274A63B89D_2/ULGbROO1OvwPmOIyaxMyIZBm5NL06Igzd4JTIaKVbLQz/Image.png)

> ### RePLUG: Retrieval-Augmented Black-box Language Model

   > ‼️**What is the difference between RePLUG and the mixture version model.**

   > 🉐 **::有没有可能更精细操作这些passage？::**

   > This paper focuses on the challenges presented by Large language Model when they are treated as black-box system, precluding further training.

   > #### Solution.

   - > ensemble method. Rather than using a mixture style of dual encoder output and get a max likelihood estimation. This method use the majority voting to determine a final result.
   - > Training the retriever by using the KL distance between output of the retriever and the majority voting from the LLM.
$$
P_{R}(d|x) = \frac{e^s(d,x)/\gamma}{\sum_{d\in D'}e^s(d,x)/\gamma}
$$

$$
Q(d|x,y) = \frac{e^P_{LM}(y|d,x)/\beta}{\sum_{d\in D'} e^P_{LM}(y|d,x)/\beta}
$$

$$
\mathcal{L} = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} KL(P_R(d|x)\| Q_{LM}(d|x,y))
$$

4. > Directions for solving these problems.

> Additionally, the people and groups that need attention.

### ::RAG as a way to reduce the back-draw of AIGC。::

   updating knowledge

   long tail data

   data leakage

   high training and inference cost

### ::The role of RAG::

   1. It helps the process of AIGC. It can reduce the cost of generationg.
      - retrieval reduce the cost of generations. retrial the old one rather than generating new context.
      - RAG helps to reduce the size of the LLM model.
      - RAG enables the LLM to generate longer content.
      - RAG may remove some steps of the AIGC

### ::Methods.::

   #### Input to the LLM at the middle stage

   #### Contribute to the final logts.

   #### Influence or omit certain generation steps.

### Augementation across multiple domains.

### ::Retriever.::

   1. Sparse.

      Text. Key value paires

      ::metrics::. TF-IDF, query likelihood, BM25.

      ::trick::. To enable efficients search, sparse retrieval typically leverage an inverted index to organize documents. **Each item in the query is mapped to a list of candicates documents.**

   1. Dense.
   2. Other

