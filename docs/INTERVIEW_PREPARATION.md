# ContextQA: 25 Technical Interview Questions & Answers
## Comprehensive Technical Q&A for Academic Presentations & AI Job Interviews

This guide contains **25 high-impact questions and answers** frequently asked in Senior NLP Engineer, Machine Learning Engineer, and Data Scientist interviews regarding Transformer architectures, Question Answering, Fine-Tuning, and practical optimization.

---

### Q1: What is the fundamental difference between Extractive and Generative Question Answering?
**English Answer**:
In Extractive QA, the model identifies and extracts the start and end indices of the answer span directly from a provided reference context passage. The vocabulary of answers is strictly constrained to contiguous substrings of the input text, completely eliminating hallucinations. In Generative QA (Abstractive), an autoregressive decoder generates free-form text token-by-token based on the question and context, allowing synthesis across disparate parts of the text but introducing a risk of factual fabrication.

**الشرح بالمصري**:
في الـ Extractive QA الموديل مش بيكتب كلام من عنده، ده زي طالب بيقرأ قطعة فهم وبيحط خط بالقلم الرصاص تحت أول كلمة وآخر كلمة في الإجابة. ده بيمنع الهلوسة تماماً (Zero Hallucination). أما الـ Generative QA فالموديل بيكتب تعبير بصياغته، وده ممكن يخليه يخترع معلومات مش موجودة في النص الأصلي.

---

### Q2: Why is BERT well-suited for Extractive QA, but GPT is primarily used for Generative tasks?
**English Answer**:
BERT is an Encoder-only Transformer trained with masked language modeling (MLM) and next sentence prediction (NSP). Its self-attention mechanism is fully bidirectional, allowing every token to attend to all preceding and following tokens simultaneously, creating rich contextualized representations of both question and passage. GPT is a Decoder-only autoregressive model that utilizes causal (unidirectional) masking, preventing tokens from attending to subsequent tokens. While causal masking is optimal for sequential text generation, bidirectional encoding is vastly superior for span boundary classification.

**الشرح بالمصري**:
الـ BERT عبارة عن Transformer Encoder بيشوف الجملة كلها في نفس الوقت (يمين وشمال) فبيقدر يفهم العلاقة بين السؤال والسياق بعمق شديد. أما GPT فهو Decoder معمول عشان يولد كلام كلمة وراء كلمة فبيشوف اللي قبله بس وميعرفش يشوف اللي بعده (Causal Masking)، وده مينفعش في استخراج إجابة دقيقة من قطعة لأنك محتاج تقرأ القطعة كلها وتفهمها قبل ما تقرر مكان الإجابة.

---

### Q3: What is Knowledge Distillation and how does DistilBERT achieve comparable performance to BERT?
**English Answer**:
Knowledge Distillation is a model compression framework where a compact "student" model is trained under the supervision of a larger "teacher" model. DistilBERT reduces BERT-base from 12 layers to 6 layers by initializing the student with alternate layers from the teacher and training it using a triple loss function: (1) distillation loss matching the teacher's soft probability distributions, (2) masked language modeling loss on hard labels, and (3) cosine embedding loss aligning the hidden states. This retains 97% of BERT's performance with 40% fewer parameters and 60% faster inference.

**الشرح بالمصري**:
تقطير المعرفة (Distillation) زي ما يكون عندك أستاذ كبير (Teacher) عنده خبرة ضخمة بيعلم تلميذ شاطر وموهوب (Student). الموديل الصغير (DistilBERT: 6 طبقات) بيتعلم مش بس من الإجابات الصح، بل بيتعلم من الاحتمالات الدقيقة اللي بيطلعها الأستاذ الكبير (BERT: 12 طبقة). النتيجة: حجم أصغر بـ 40% وسرعة أكبر بـ 60% مع الاحتفاظ بـ 97% من الدقة!

---

### Q4: Why do we tokenize question and context together as a single sequence rather than passing them into two separate encoders?
**English Answer**:
Concatenating question and context into `[CLS] Question [SEP] Context [SEP]` enables cross-attention across all layers of the Transformer. Every token in the question directly attends to every token in the context and vice-versa through the multi-head self-attention mechanism. If encoded separately (e.g., via dual encoders / bi-encoders), token-level cross-interactions are lost, severely degrading span boundary accuracy.

**الشرح بالمصري**:
لو دخلت السؤال لوحده والقطعة لوحدها في موديلين منفصلين، الكلمات مش هتعرف تتكلم مع بعضها. لكن لما ندمجهم مع بعض ويفصل بينهم `[SEP]`، كل كلمة في السؤال بتعمل Self-Attention مع كل كلمة في القطعة في كل طبقة من طبقات الموديل، فيقدر يحدد مكان الإجابة بدقة متناهية.

---

### Q5: How is the Question Answering head structured mathematically in Hugging Face Transformers?
**English Answer**:
The QA head is a linear projection layer applied directly to the final hidden representations:
$$\mathbf{H} \in \mathbb{R}^{L \times d}$$
Where $L$ is sequence length and $d$ is the hidden dimension (768 for DistilBERT). The linear layer projects $\mathbf{H}$ to two scalar logits per token:
$$\mathbf{L} = \mathbf{H}\mathbf{W}^T + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{2 \times d}, \quad \mathbf{b} \in \mathbb{R}^2$$
Splitting $\mathbf{L}$ yields $\mathbf{s} \in \mathbb{R}^L$ (start logits) and $\mathbf{e} \in \mathbb{R}^L$ (end logits). Softmax is computed across the sequence length to produce independent probability distributions for start and end positions.

**الشرح بالمصري**:
رأس الموديل (QA Head) عبارة عن Linear Layer بسيطة جداً بتأخذ المخرجات بتاعة كل توكن (Vector طوله 768) وتحوله لرقمين اثنين فقط: Start Logit (احتمالية إنه يكون أول حرف في الإجابة) و End Logit (احتمالية إنه يكون آخر حرف).

---

### Q6: What loss function is used to train Extractive Question Answering models?
**English Answer**:
The training loss is the average of two Cross-Entropy losses:
$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{start}} + \mathcal{L}_{\text{end}} \right)$$
$$\mathcal{L}_{\text{start}} = -\log P(y_{\text{start}}), \quad \mathcal{L}_{\text{end}} = -\log P(y_{\text{end}})$$
Where $y_{\text{start}}$ and $y_{\text{end}}$ are the ground-truth token indices of the target span. When an answer is outside a sliding window chunk, the target label points to index 0 (`[CLS]`).

**الشرح بالمصري**:
بنستخدم Cross-Entropy Loss عادي جداً، بس بنحسبه مرتين: مرة لمكان البداية (Start) ومرة لمكان النهاية (End)، وبعدين بناخد المتوسط بتاعهم. لو الإجابة مش موجودة في الجزء ده من النص، الـ Target بيكون التوكن رقم 0 اللي هو `[CLS]`.

---

### Q7: Why can't we simply predict character indices directly instead of token indices?
**English Answer**:
Transformers operate on discrete token embeddings generated by subword tokenizers. The hidden states correspond directly to tokens, not raw characters. Attempting character-level index prediction would require either character-level tokenization (which destroys semantic chunking and creates excessively long sequences) or complex continuous regression heads, which suffer from poor convergence compared to discrete token classification. Offset mapping bridges the gap by projecting predicted token spans back to exact character slices.

**الشرح بالمصري**:
الموديل بيعالج توكنز مش حروف. لو خليناه يتوقع رقم الحرف مباشرة هيبقى Regression Task معقدة وغير دقيقة، ولو خليناه يقسم الكلام لحروف مفردة المتسلسلة هتبقى طويلة جداً ومملة على كارت الشاشة. الأفضل والأدق هو التوقع على مستوى التوكنز، ونستخدم الـ Offset Mapping عشان نرجع لرقم الحرف الأصلي.

---

### Q8: What is Offset Mapping and why is `is_fast=True` necessary for Hugging Face tokenizers?
**English Answer**:
`offset_mapping` is a list of integer pairs `(start_char, end_char)` indicating the exact character slice in the original text represented by each subword token. It is generated during tokenization. Fast tokenizers (written in Rust under the `tokenizers` library) compute and store these character offsets in linear time with zero Python overhead. Standard Python-based tokenizers do not compute offset mappings natively, making alignment impossible for high-throughput SQuAD processing.

**الشرح بالمصري**:
الـ Offset Mapping هو الخريطة اللي بتقولك: التوكن رقم 5 ده بيمثل الحروف من 24 لـ 31 في النص الأصلي. والـ Fast Tokenizers المكتوبة بلغة Rust بتعمل الحسابات دي بسرعة الصاروخ، وهي الوحيدة اللي بتدعم خاصية `return_offsets_mapping=True`.

---

### Q9: Explain the Sliding Window technique and why `doc_stride` is critical.
**English Answer**:
Transformers have a quadratic attention complexity $\mathcal{O}(L^2)$ and hard positional embedding limits (typically 512 tokens). When passages exceed `max_seq_length = 384`, truncating would permanently discard answers in later paragraphs. The sliding window splits long contexts into overlapping chunks. `doc_stride` specifies the token overlap (e.g., 128 tokens) between consecutive windows, guaranteeing that any answer span located near the boundary of one chunk appears fully preserved in the subsequent chunk.

**الشرح بالمصري**:
لما يكون المقال طويل، مينفعش نقص الباقي ونرميه لأن الإجابة ممكن تكون في الآخر. بنقطعه لقطع (Windows) متداخلة. التداخل (Stride) ده بيضمن إن لو الإجابة جت على حافة القطعة الأولى وماتلحقتش كاملة، هتيجي في القطعة اللي بعدها مرتاحة وفي نص الكلام.

---

### Q10: How do you handle unanswerable chunks during sliding window training?
**English Answer**:
In SQuAD v1.1, every question has an answer in the overall passage. However, when a passage is split into multiple sliding window features, some features will not contain the answer. For any chunk where the ground-truth span is partially or completely outside the chunk's boundaries, we set `start_position = 0` and `end_position = 0` (pointing to the `[CLS]` token). The model learns to output high logits on `[CLS]` whenever a passage snippet lacks the necessary information.

**الشرح بالمصري**:
لو قسمنا المقال لـ 3 أجزاء، والإجابة في الجزء الأول بس؛ الأجزاء الثاني والثالث الموديل لازم يعرف إن مفيش فيهم إجابة. فبنديله الـ Target بتاعهم عند التوكن رقم 0 اللي هو `[CLS]`. ده بيعلم الموديل يقول: "أنا مش شايف إجابة هنا".

---

### Q11: During inference, how do you prevent the model from predicting invalid spans?
**English Answer**:
Post-processing enforces three structural constraints on candidate spans $(i, j)$:
1. **Context Filtering**: Tokens must belong to the context (i.e., `sequence_ids == 1`), excluding question tokens and special tokens.
2. **Monotonicity**: $j \ge i$ (end position cannot precede start position).
3. **Length Constraint**: $(j - i + 1) \le \text{max\_answer\_length}$ (e.g., 30 tokens).
Any pair violating these conditions is masked or discarded before computing span scores $\text{score} = s_i + e_j$.

**الشرح بالمصري**:
بنفلتر الإجابات الرياضياً:
1. ممنوع يختار كلمة من السؤال أو رمز زي `[SEP]`.
2. ممنوع النهاية تيجي قبل البداية ($j \ge i$).
3. ممنوع طول الإجابة يزيد عن 30 كلمة عشان ميجبلكش نص المقال كإجابة!

---

### Q12: Why is raw softmax across logits uncalibrated in deep neural networks?
**English Answer**:
Modern deep networks with batch normalization, layer normalization, and cross-entropy loss are notoriously overconfident. Taking $\text{softmax}(\mathbf{z})$ yields values in $[0, 1]$, but these do not represent true empirical probabilities $P(\hat{y} = y | \hat{P} = p) = p$. Temperature scaling or Platt scaling is required to calibrate model outputs. In ContextQA, we document the confidence score as an uncalibrated relative ranking indicator rather than a true Bayesian probability.

**الشرح بالمصري**:
الـ Softmax بيطلع أرقام بين 0 و 1، والناس بتفتكرها نسبة مئوية حقيقية للتأكد. لكن شبكات الـ Deep Learning بطبيعتها "واثقة في نفسها بزيادة" (Overconfident)، فممكن تطلع 99% وهي بتخرف! عشان كدة هندسياً لازم نوضح للمستخدم إن ده مؤشر ثقة نسبي مش احتمال إحصائي مؤكد.

---

### Q13: What is the Exact Match (EM) metric in SQuAD v1.1?
**English Answer**:
Exact Match is a strict binary metric that assigns 100.0% if the predicted string matches the ground-truth string character-for-character after standard text normalization (lowercasing, punctuation stripping, and article removal: 'a', 'an', 'the'), and 0.0% otherwise. When multiple valid ground-truth answers exist, EM takes the maximum score across all references.

**الشرح بالمصري**:
الـ Exact Match معناه الإجابة طلعت مطابقة 100% بدون أي كلمة زيادة أو ناقصة بعد ما نشيل علامات الترقيم وحروف التعريف الإنجليزية (a, an, the). لو الموديل زود كلمة واحدة، بياخد صفر!

---

### Q14: How is Token-Level F1 calculated in SQuAD, and why is it preferred over Exact Match?
**English Answer**:
Token-level F1 treats the prediction and ground-truth as bags of normalized words. Precision is the ratio of shared words to total predicted words; Recall is the ratio of shared words to total ground-truth words:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
F1 is preferred because it rewards partial overlap. For example, if the ground truth is "George Washington" and the model predicts "President George Washington", EM is 0% while F1 is 80%, providing a continuous, informative signal of semantic accuracy.

**الشرح بالمصري**:
الـ F1 بيحسب الكلمات المشتركة بين تنبؤ الموديل والإجابة الصح. لو الإجابة "أحمد زويل" والموديل قال "العالم أحمد زويل"، بياخد 80% في الـ F1 بدل ما ياخد صفر في الـ EM. عشان كدة هو أعدل وأقرب لتقييم الفهم الحقيقي.

---

### Q15: How do you optimize training on an NVIDIA RTX 3060 Laptop GPU with only 6GB VRAM?
**English Answer**:
Four engineering strategies were employed:
1. **FP16 Mixed Precision**: Uses 16-bit half-precision floats for activations and weights, halving memory footprint and leveraging Tensor Cores.
2. **Gradient Accumulation**: Uses a physical batch size of 8 with 2 accumulation steps, delivering an effective batch size of 16 without VRAM overhead.
3. **Sequence Length Tuning**: Caps length at 384 tokens with stride 128 rather than 512, saving significant quadratic memory.
4. **PyTorch Dynamic Allocation**: Runs garbage collection and avoids caching redundant intermediate tensors.

**الشرح بالمصري**:
1. شغلنا الـ **FP16 Mixed Precision** فاستهلكنا نص الذاكرة وسرعنا الحسابات بفضل الـ Tensor Cores.
2. استخدمنا **Gradient Accumulation** عشان ندرب بـ Batch size 8 ونجمع الـ Gradients كأننا بندرب بـ 16 أو 32 بدون ما كارت الشاشة يفرقع OOM.
3. خلينا طول المتسلسلة 384 بدل 512 توكن.

---

### Q16: What is the purpose of Learning Rate Warmup and Linear Decay in AdamW?
**English Answer**:
In early fine-tuning iterations, pre-trained weights are well-tuned while the newly initialized QA head weights are random. Large gradient steps early on would cause catastrophic forgetting in the backbone. A linear warmup gradually ramps the learning rate from 0 to peak ($\text{lr} = 3 \times 10^{-5}$) over the first 10% of steps, stabilizing the head. Subsequently, linear decay anneals the learning rate towards 0, ensuring smooth convergence to a sharp local minimum.

**الشرح بالمصري**:
في بداية التدريب، الـ Transformer بيكون متعلم لغة كويس جداً لكن رأس السؤال والجواب (QA Head) بيكون متسفر عشوائياً. لو دخلت عليه بـ Learning Rate عالي فجأة، هيمسح اللي اتعلمه في الـ Pre-training (Catastrophic Forgetting). الـ Warmup بيسخن بالراحة في أول 10% من الخطوات، وبعدين يقلل السرعة تدريجياً لغاية ما يستقر الموديل.

---

### Q17: What is Catastrophic Forgetting and how do we prevent it during QA fine-tuning?
**English Answer**:
Catastrophic forgetting occurs when a pre-trained neural network overwrites its foundational language representations while optimizing for a narrow downstream task. It is mitigated by: (1) using very small learning rates ($2 \times 10^{-5}$ to $5 \times 10^{-5}$), (2) using AdamW with weight decay (0.01) to regularize parameter magnitude, and (3) training for very few epochs (2 to 3 epochs on SQuAD).

**الشرح بالمصري**:
النسيان الكارثي هو إن الموديل ينسى قواعد الإنجليزي وفهم النصوص اللي اتعلمها في شهور، عشان مركز في مهمة SQuAD بس. بنمنع ده بإننا نستخدم Learning Rate صغير جداً (3e-5) وتدريب لعدد Epochs قليل (2 لـ 3 بس).

---

### Q18: What is the difference between WordPiece tokenization (BERT) and Byte-Pair Encoding (RoBERTa)?
**English Answer**:
WordPiece forms subwords by maximizing the language model likelihood of the training corpus when adding candidate merges. It prefixes continuation subwords with `##`. BPE (and Byte-level BPE used in RoBERTa) starts from raw bytes and iteratively merges the most frequent adjacent character/byte pairs. Byte-level BPE requires no pre-tokenization or vocabulary additions for foreign scripts or emojis, encoding any arbitrary UTF-8 string without unknown tokens (`[UNK]`).

**الشرح بالمصري**:
الـ WordPiece (بتاع BERT) بيستخدم علامة `##` للمقاطع الداخلية وبيدور على المقاطع اللي بتعظم احتمالية الكوربس. أما BPE في RoBERTa بيتعامل على مستوى الـ Bytes (البايتات المباشرة)، وده بيخليه يقدر يقفل على أي رمز أو إيموجي أو لغة في العالم وميطلعش `[UNK]` أبداً.

---

### Q19: Why does RoBERTa generally outperform BERT on SQuAD v1.1?
**English Answer**:
RoBERTa (Robustly Optimized BERT Approach) improves upon BERT through: (1) training on 160GB of text (10x larger than BERT), (2) training with larger mini-batches (up to 8k sequences) and longer steps, (3) removing the Next Sentence Prediction (NSP) task which was found to hurt downstream performance, and (4) dynamically changing masked token patterns during pre-training epochs.

**الشرح بالمصري**:
موديل RoBERTa اتدرب على داتا أكبر 10 مرات، بـ Batch size ضخم، وشالوا منه مهمة NSP اللي أثبتت الأبحاث إنها ملهاش لازمة، وخلو الـ Masking ديناميكي بيتغير في كل لفة، فطلع أذكى وأقوى في استخراج الإجابات.

---

### Q20: How do you evaluate latency in production NLP systems?
**English Answer**:
Inference latency must be measured as the end-to-end elapsed wall-clock time per request:
$$\text{Latency} = t_{\text{tokenize}} + t_{\text{forward\_pass}} + t_{\text{postprocess}}$$
Benchmarking requires warming up the GPU with several dummy passes, measuring percentiles (P50, P95, P99) across hundreds of representative inputs, and reporting throughput in queries per second (QPS).

**الشرح بالمصري**:
في الإنتاج مش بنقيس سرعة الموديل بس؛ بنقيس الرحلة كاملة: وقت التقطيع (Tokenization) + وقت تشغيل الموديل على كارت الشاشة + وقت استخراج النص من الـ Logits. وبنهتم جداً بالـ P95 و P99 عشان نضمن مفيش مستخدم يقف مستني ثواني طويلة.

---

### Q21: What are the primary failure modes of Extractive QA models?
**English Answer**:
Analysis reveals four dominant error categories:
1. **Boundary Shifts**: Correct entity identified, but includes or omits modifying articles or prepositions (e.g., "in 1885" vs "1885").
2. **Entity Confusion**: Picking an alternative entity mentioned in the same context (e.g., selecting the founder's father instead of the founder).
3. **Complex Syntactic Questions**: Failing to resolve multi-hop or negation relationships within complex paragraphs.
4. **Context Truncation / Out-of-Window**: Answer split across sliding window boundaries without sufficient overlap.

**الشرح بالمصري**:
أشهر 4 أخطاء بيقع فيها الموديل:
1. يزود أو ينقص حرف جر أو أداة تعريف (Boundary Shift).
2. يلخبط بين اسمين مذكورين في نفس السطر (Entity Confusion).
3. ميفهمش جمل النفي أو الأسئلة المركبة اللي محتاجة خطوتين استنتاج.
4. الإجابة تتقطع بسبب سوء حجم الـ Window.

---

### Q22: What happens if the answer is not present in the context? How does SQuAD v2.0 differ from v1.1?
**English Answer**:
SQuAD v1.1 assumes every question is answerable from the context. SQuAD v2.0 introduces over 50,000 unanswerable questions that look plausible. In SQuAD v2.0, models must predict whether an answer exists by computing the score difference between the best candidate span and the null answer score at index 0 (`[CLS]`):
$$\Delta = \text{score}_{\text{best}} - \text{score}_{\text{null}}$$
If $\Delta < \tau$ (threshold), the model outputs "unanswerable".

**الشرح بالمصري**:
في SQuAD v1.1 الموديل عارف إن فيه إجابة أكيدة في الفقرة. أما في SQuAD v2.0 حطوا أسئلة خبيثة ملهاش إجابة، والموديل لازم يقارن سكور أفضل إجابة مع سكور الـ `[CLS]`، ولو ملقاش إجابة قوية يقول بثقة: "السؤال ده ملوش إجابة في القطعة".

---

### Q23: Why do we set `dataloader_num_workers=0` on Windows systems?
**English Answer**:
In PyTorch on Windows, multiprocessing uses `spawn` rather than `fork` (used in POSIX/Linux). Spawning multiple workers creates entirely new Python sub-processes that re-import the main script and duplicate CUDA contexts, frequently causing deadlocks or memory leaks unless wrapped strictly in `if __name__ == '__main__':`. Setting `dataloader_num_workers=0` keeps data collation in the main process, ensuring 100% stability.

**الشرح بالمصري**:
في الويندوز، بايثون مش بيعمل Fork للعمليات زي لينكس، بل بيعمل Process جديدة تماماً (Spawn) بتعيد تحميل كل المكتبات من الأول وبتعمل مشاكل وتعليق في الـ CUDA. فاختيار 0 على ويندوز بيضمن استقرار التدريب بدون كراش.

---

### Q24: What is the computational complexity of the Self-Attention mechanism?
**English Answer**:
Self-Attention scales quadratically with sequence length:
$$\mathcal{O}(L^2 \cdot d)$$
Where $L$ is sequence length and $d$ is embedding dimension. This quadratic bottleneck $\mathcal{O}(L^2)$ arises from computing the attention score matrix $\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times L}$. This is the exact reason why we limit $L=384$ and use sliding windows rather than passing 2,000 tokens at once.

**الشرح بالمصري**:
الـ Self-Attention حسابه بيكبر تربيعياً مع طول النص $\mathcal{O}(L^2)$. لو النص طوله 100 توكن، بيعمل 10,000 عملية، لكن لو طوله 1,000 توكن بيعمل 1,000,000 عملية! عشان كدة مينفعش نرمي مقال كامل في الموديل، ولازم نقسمه لـ Windows بطول 384.

---

### Q25: How would you scale this Extractive QA system to search across millions of documents?
**English Answer**:
A production enterprise QA system follows a **Retriever-Reader Architecture**:
1. **Retriever (Bi-Encoder / Dense Passage Retrieval / BM25)**: Scans millions of documents using vector search (e.g., FAISS, Qdrant) or inverted indices to quickly retrieve the top 5-10 most relevant passages in milliseconds.
2. **Reader (Cross-Encoder / ContextQA DistilBERT)**: Takes the top retrieved candidate passages along with the user's question, runs our extractive pipeline, and pinpoints the exact answer span.

**الشرح بالمصري**:
لو عندك مليون وثيقة مينفعش تعدي عليهم بـ DistilBERT واحد واحد لأن ده هياخد ساعات. بنعمل معمارية اسمها **Retriever-Reader**:
1. **Retriever**: بيبحث بسرعة في المليون وثيقة ويجيب أفضل 5 فقرات في 20 ميلي ثانية (باستخدام Vector DB زي FAISS).
2. **Reader**: هو موديل الـ ContextQA بتاعنا، بيمسك الـ 5 فقرات دول ويفصصهم ويطلع الإجابة والسطر والكلمة المحددة للمستخدم!
