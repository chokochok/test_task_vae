# Task Definition

## TEST TASK: VAE FOR IMAGE GENERATION

### Goal
Train a Variational Autoencoder that can:
- reconstruct input images well, and
- generate plausible new images by sampling from the latent space.

You must demonstrate you can debug typical VAE failure modes and provide quantitative + qualitative evidence.

### Dataset
Pick one dataset below:
- CIFAR-10 (32×32, RGB);
- CelebA (crop/resize to 64×64, RGB);
- Fashion-MNIST (28×28, grayscale).

You may use torchvision dataset loaders if available.

### Requirements

#### Implement a convolutional VAE
- Build a VAE with Encoder, Decoder, Reparameterization.
- You must write down and implement the ELBO loss.

#### Make it train stably
Add at least two of the following (and explain why you chose them):
- KL warm-up / annealing schedule
- β-VAE (β ≠ 1)
- Gradient clipping
- Free bits
- Better decoder likelihood (choosing a likelihood model that matches image data better)

If you diagnose any training issues, you must explicitly show the evidence.

#### Evaluation
Provide qualitative and quantitative evaluations.
Also include a "failure gallery", showing 10–20 bad samples/reconstructions and hypothesize why.

### Engineering deliverables
A GitHub repository that contains (minimal requirements):
- `train.py` that runs end-to-end;
- `evaluate.py` producing metrics + image grids;
- reproducible config (JSON/YAML/etc.);
- clear folder structure;
- saved artifacts: checkpoints, generated images, training curves (loss, recon term, KL term);
- `README.md` with a report.

### Candidate report (1–2 pages max)
MUST HAVE:
- dataset & preprocessing;
- architecture summary + parameter counts;
- loss design + why;
- metrics + grids;
- top issues encountered and fixes;
- next steps (what you'd do to improve sharpness/diversity).

### What we will grade
- **Correctness (30%)**: proper ELBO, reparameterization, stable training
- **Generative quality (25%)**: sample realism/diversity + your metric
- **Debugging maturity (20%)**: you identify issues and fix them with justified changes
- **Engineering (15%)**: reproducible runs, clean code, clear outputs
- **Communication (10%)**: concise report with evidence and trade-offs

### Execution Time
You will have up to 2 days to complete the task.


я провів тренування на кожному датасеті по 50 епох
що я отримав: для фешн і селеба реконструкція і генерацію +- нормальні, для сіфар погані (дуже мутні, жодних об'єктів не видно)
також по візуальному аналізі я замітив, що найгірше модель справляється з зображеннями, де є багато контрастних дателей/ліній (особливо це замітно на селеба, де є шоломи велосипедистів або текст або різнокольорові контрастні лінії на фоні, для фешина найгіршим випадкам просто не вистачає контрасту і чіткості коли одяг має складну форму (полоски на одязі або туфлі з вирізами))

щодо метрик: дуже сильно коливається лос тренування, флуктуації сильні усіх метрик і також kld росте (що на трейні що на валі), хоча інші метрики всетаки по чучуть зменшуються по медіані

експериментую на фешнмністі
найкращий лр 0.001

додав щоб зменшити флуктцації
weight_decay: 1e-5        # 0.0 щоб вимкнути
betas: [0.9, 0.999]       # Стандартні. Для стабільності можна [0.5, 0.9]

use_scheduler: true       # Вмикач шедулера
scheduler_patience: 3     # Скільки епох чекати
scheduler_factor: 0.5     # На скільки зменшувати LR

додаємо нормалізацію лосу, клампінг логвару і Б-вае < 1

проблема: при використанні анілу вал лос має історичний мінімум в перших епохах
змінюємо логіку збереження на сейв ласт і кожні 10 епох

при використанні аннілу 0.1 якість рандомних семплів дуже сильно впала, хоча якість реконструкції покращилася

оскільки лос і рекон є близькі (10, 8) не має сенсу використовувати Б менше 1

експериментально підтверджено, що без к-неалінгу рекон вище ніж з на тих самих епохах, хоча лос вирівнявся
0, 9, 10, 11

12 (0.5), 13 (0), 14 (0.1), 15 (0.01)

вирішив нормалізувати лоси на кільіксть пікселів, щоб між датасетами була відтворюваність +-



2.0 почистив код, забрав нормалізацію і почав тестування знову
vae_annealing_clip_fashion_mnist
0 тренування без оптимізацій, тільки нові хід діменшини
1 gradient clip 0.1
2 gradient clip 0.5

clip зменшив всі лоси, 0.5 покращив, але рекон вищий, кл нижчий ніж 0.1
залишаємо 0.1, тому що 0.5 заставляє модель лінуватися ігноруючи деталі картинки

3 залишаємо 0.1, додаєму анніалінг 0 до 1 10 епох
4 аннаіл до 0.1
5 аннаіл 0.8
6 просто 0.8 без аннаілу
7 аннаіл 0.5 до 0.8
8 просто 3.0
9 просто 1.3

vae_annealing_clip_final_fashion_mnist
0 згенеровано фінальний результат для візуалізації

vae_annealing_clip_cifar10
0 нові діменшини
1 кліп 0.1, анніалінг 1.0
2 анніалінг 0.5
3 0.1
4 латент дім 256 бета 0.25
зображення дуже мильні, важко побачити якийсь обєкт
збільшуємо хіден діменшини
5 [128, 256, 512] бета 0.25 10 епох
модель не встигає навчитися робити реконструкції
6 ставимо бета 0 щоб перевірити чи архітектруа взагалі вміє реконструювати
7 додаємо кл вармап 5 бета 0.1 аніл 5
8 7 але бета 0.5

vae_annealing_clip_final_cifar10
0 фінальний результат (мильно, є багато місця для покращення)

vae_annealing_clip_celeba
0 нові деменшини, без оптимізацій
кл дуже скаче, але реконструкція і генерація виглядають нормально (у всіх +- однакове лице)
1 кліп 0.1 анніал 2 епох вармап 2
вдалося досягти кращої реконструкції на початку, що додало індивідуальності лицям

vae_annealing_clip_final_celeba
0 бета 0.9 для кращої ідентичності

cifar10 opt
1 update config wider rather deeper
2 SiLU + L1 loss
3 ResNet (ResidualBlock + Upsample)
4 up kl_end, resnet false
5 more kl
6 kl 0.5, resnet true
7 kl 0.8
8 increase latent and hidden
9 no anneal kl 1.0
10 11 test with different sizes
12 vgg test
13 14 vgg with resnet
