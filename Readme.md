
# ML01 Machine Learning, UTSEUS, Shanghai University

## Language

English. For everything.

## Where and When

### Tencent Meeting

For each session, please always join Tencent Meeting (VooV Meeting):
- Room ID：958 9491 5777

### Laptop

For each session, please bring your own Laptop!

For Thursdays' practice sessions, please bring your headphone as well, because you will watch videos.

### Monday (Lectures and Continuous assessments)

- 20:00 - 21:40
- B313

### Wednesday (Exercise sessions)
- 10:00 - 11:40
- B313

### Thursday (Practice sessions)
- 13:00 - 16:40
- B315


## Lectures (Monday)

### Week 1
- Machine Learning overview

### Week 2
- Linear Regression

### Week 3
- Logistic Regression (for classification)

### Week 4
- Neural networks

During the class,
we will play a little bit with 
Tensorflow Playground:
- https://playground.tensorflow.org

AFTER the class, please watch those videos very carefully:
- https://www.youtube.com/watch?v=aircAruvnKk
- https://www.youtube.com/watch?v=IHZwWFHWa-w
- https://www.youtube.com/watch?v=Ilg3gGewQ5U
- https://www.youtube.com/watch?v=tIeHLnjs5U8

### Week 5
- Building a Machine Learning web app

### Week 6
- Model selection

### Week 7
- CNN
    - for image classification
    - for image segmentation

### Week 8
- GAN

### Week 9
- AutoEncoder

### Week 10
- DQN

## Continuous assessment (Monday)

Tests will take place on Mondays (Week 2, Week 4, Week 6, Week 8).

Each test falls in the topic of its previous week, with some extensions (e.g. some more math).

You are recommended to read materials provided by prof ahead of time, to maximize your chance of success.

In total, 4 tests will be conducted.

Tests are on paper, with book closed, no Internet, no electronic device, no discussion with classmates, no asking prof questions.

After each test, feel free to forget everything that you have learned for test preparation. Because your intuition has already been developped and will stay with you. After you have experienced all this, you gain more confidence on youself and would be more open to new challenges. And that's the most important thing.


### Week 2

Materials to read before test:
- all jupyter notebooks for  lectures and exercises
- https://www.t-ott.dev/2021/11/24/animating-normal-distributions
- https://demonstrations.wolfram.com/TheBivariateNormalDistribution/
- https://online.stat.psu.edu/stat505/lesson/4/4.2
- https://github.com/features/actions
- https://docs.github.com/en/actions/quickstart
- https://github.blog/2022-02-02-build-ci-cd-pipeline-github-actions-four-steps/
- https://resources.github.com/ci-cd/
- https://github.com/readme/guides/sothebys-github-actions
- https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
- https://www.ssh.com/academy/ssh-keys

Test (30 min):
- Python list
- Numpy slicing, numpy broadcast
- **Math**: Bivariate Gaussian Distribution
- ssh key
- CI/CD, GitHub Actions


### Week 4

Materials to read before test:
- all jupyter notebooks for  lectures and exercises
- https://www.bilibili.com/video/BV1SY4y1G7o9/


Test (30 min):
- Linear Regression implementation from scratch
- Logistic Regression implementation from scratch
- **Math**: gradient descent for linear regression and logistic regression
- GitHub Pull Request (GitHub workflow)
- git conflict resolving
- git merge v.s. git rebase


### Week 6
Materials to read before test:
- all jupyter notebooks for lectures and exercises

Test (30 min):
- Neural network implementation from scratch (1/2)
- **Math**: activation functions
- **History/Culture**: History of Neural Network/Deep Learning

### Week 8

Materials to read before test:
- all jupyter notebooks for  lectures and exercises



Test (30 min):
- Neural network implementation from scratch (2/2)
- **Math**: backpropagation algorithm
- **History/Culture**: Major Research and Application breakthroughs of deep learning in recent years


## Exercise sessions (Wednesday)

Most exercises will correspond to lecture topics, with some extensions.

### Week 1

Starting from this session, we will use Jupyter Notebook

Please install Python, VS Code, and, ideally, you should be able to use Google Colab and GitHub.

Make sure you have a seamless Internet connection to those websites.

Exercise:
- Python
- Numpy
- Pandas

### Week 2

Make sure that you can run our Jupyter Notebooks on VS Code.

Also, make sure you have access to GitHub, Google and YouTube.


Exercise:
- Linear Regression from scratch
    - [Math-heavy approach](https://www.analyticsvidhya.com/blog/2021/06/getting-started-with-machine-learning%E2%80%8A-%E2%80%8Aimplementing-linear-regression-from-scratch/)
    - Math-light approach (Gradient Descent)
- https://towardsdatascience.com/coding-linear-regression-from-scratch-c42ec079902
- https://github.com/tugot17/Linear-Regression-From-Scratch
- https://debuggercafe.com/implement-simple-linear-regression-from-scratch/
- https://www.kaggle.com/code/kennethjohn/linear-regression-from-scratch
- https://www.kaggle.com/code/fareselmenshawii/linear-regression-from-scratch

### Week 3

Exercise:
- Logistic Regression from scratch    
    - Math-heavy approach
    - Math-light approach (Gradient Descent)

### Week 4

### Week 5

### Week 6


### Week 7

### Week 8

### Week 9

### Week 10

## Practice sessions (Thursday)

### Week 1 - Week 2

Tutorial (with videos):
- https://gitee.com/lundechen/static_website_with_go_hugo

The main goal of this tutorial is **NOT** to teach you Web Technology, but to walk through the main steps for building a static website, and to learn to use, in the meanwhile:
- Git (add, commit, push, pull, checkout, rebase, merge, conflict resolving)
- GitHub (GitHub pull request, GitHub actions)

Week 1:
- Finish deploying the website on GitHub Pages (Video 1 - 5).
- Send the url of your GitHub Pages to the WeChat group when you finish.

Week 2:
- GitHub workflow (Video 6 - 8).
- Each group two students.
- Please install GitLens (VS Code extension).
- Send the url of your GitHub Repositories to the WeChat group when you finish (three urls, for each group of two).

#### Pro tips

用 Windows 的同学，可以按照这个教程，安装 posh-git
- https://gitee.com/lundechen/hello#9-optional-git-branchstatus-indication-on-terminal

Corresponding tutorial video：
- https://www.bilibili.com/video/BV1cq4y1S7Be/ （starting from the 6th minute of the video）

同时 windows 建议使用 Windows Terminal

主要是为了这个：

![](img/posh-git.png)

If you are on MacOS/Linux, you can install *oh my zsh* instead.

同时 windows 建议使用 Windows Terminal

### Week 3 

#### Task 1: reveal.js
- Follow the video
    - Create a repo named `cv`
    - Deploy the website as `https:<YOUR-GITHUB-ACCOUNT-NAME>.github.io/cv`
- Go to reveal.js official website, and try out different features
    - [demo](https://revealjs.com/demo), with its corresponding [source code](https://github.com/hakimel/reveal.js/blob/master/demo.html)
    - code highlight
    - image background
    - animation
    - transition
- For each of those different features
    - create a seperate GitHub repository 
    - therefore, you end up with multiple `remote`s on your local repository
- Send the URLs of your reveal.js websites to the WeChat group when you finish

#### Task 2: Deploy the reveal.js/GoHugo websites on Tencent Static Website Hosting Service, or AWS S3
- Send the URLs of your Tencent/AWS S3 websites to the WeChat group when you finish


#### Pro tips

Emoji HTML code:
- https://www.quackit.com/character_sets/emoji/emoji_v3.0/unicode_emoji_v3.0_characters_all.cfm


Make sure you have a Tencent Cloud account, and an AWS account with a Credit Card bound to it.

For a credit card, you might need help from a friend living now in foreign countries. I am not sure a Visa card in China will do or not.

For AWS, it will cost 1$, and then everything is basically free for one year.


### Week 4 

- Test Driven Programming
    - Tutorial: https://open-academy.github.io/machine-learning/assignments/get-started.html
    - Video - Chinese version: https://www.bilibili.com/video/BV1uW4y1s7Ci
    - Video - English version: https://www.bilibili.com/video/BV1nM41167j9
- GitHub Classroom

### Week 5 - Week 7

Tutorial (with videos):
- https://gitee.com/lundechen/machine_learning_web_app

Week 4:
- Local deployment.
- Data Augmentation (no tutorial from prof.) for better performance.
    - For Data Augmentation, follow this: https://open-academy.github.io/machine-learning/assignments/ml-fundamentals/ml-overview-mnist-digits.html


Week 5:
- Cloud deployment
    - for the ML web app
    - as well as for GoHugo and reveal.js website.
- GitHub WebHook (no tutorial from prof., things are to be done by students).

Week 6:
- Docker deployment, for the ML web app, as well as for GoHugo and reveal.js website.
- Send the url of your ML web app to the WeChat group when you finish.

### Week 8

AWS SageMaker

### Week 9 - 10

Machine learning app next.js/react, AWS Amplify

Team work, IAM etc. 

Video to be made with Zhu Xinning.

Basically, it will be the AWS Amplify/next.js version of:
- https://gitee.com/lundechen/machine_learning_web_app

Students will also learn how to manager resource access with IAM control, because every two students will pair up and work together.

## Project

### Forming groups

Each group 3 students.

Forming groups:
- https://docs.qq.com/doc/DT2xqVHphanhGUWpR

At most ONE group could have 2 or 4 students, provided that `N_Student % 3 != 0`.


### Get inspired
Streamlit Gallery etc.

### Implementation

Your machine learning web application can be based on streamlit, flask, next.js, tensorflow.js or any other framework. It should be deployed on the cloud (Tencent/Alibaba/Google/Microsoft Cloud).

You can use chatgpt or gpt4, if you have the API key.

You can use AI APIs from Baidu/Tencent/Alibaba/Amazon/Google etc., for example:
- https://cloud.tencent.com/product/ai-class

Apply for a domain name if necessary, e.g. [http://an-interesting-ml-app.com](http://an-interesting-ml-app.com).

As an alternative, a WeChat miniprogram is OK as well.

Use emojis or fontawesome/bootstrap icons. 

You code should be open-sourced and hosted on GitHub.

### What's expected of your video
- Length of video \>= 20 min
- You video should include those contents:
    - General presentation
    - Where do you get inspirations from for coming up with the idea of your project
    - How to use your ML Web app
    - How did you implement your app
    - How did you deploy your app
- If possible, make it fun. 
- If possible, make it fancy. 
- If applicable, include an ethics analysis of your project in the video.
- If applicable, include an social impact analysis of your project  in the video.
- If applicable, include an market analysis of your project  in the video.
- If applicable, include an ecology analysis of your project  in the video.
- And yes, your video should be presented in English. 

### Submission of your work

1. Create a folder, in which you put:
    - the video
    - the source code
    - a txt/markdown file indicating 
        - what's the task of each team member
        - the estimated workload/contribution percentage of each team member 
    - a txt/markdown file indicating 
        - the URL of your GitHub repository for hosting your code
        - prof will check the commit history of your GitHub repo to see how each team member is contributing  
1. Zip the folder
1. Upload the zip file to Google Drive
1. Send the sharing link to the prof, by PRIVATE WeChat or by Email
    - Therefore, in the WeChat/Email message, there are no attached files, just an Google Drive URL.

For each team, just one submission of the work is necessary, by one member of your team.

Deadline for submission:
- The second Friday of the 14 days of Exam Weeks of SHU, 23:59.

### For best projects
- Best projects might be hosted on [http://lunde.top](http://lunde.top), to inspire future projects.
- Best projects' videos will be included on Lunde Chen's bilibili channel.
- Lunde Chen might invite you to participate in innovation competitions with your projects.

### Last but not least
Your app should be legal, ethical.

## Score

Denoting your Continuous assessment score as `T`, your project score as `P`,
your final score will be 

```python
max(P, 0.4 * T + 0.6 * P)
```

The average of all `T`s of all students will be equal to the average of all `P`s.

### Distribution of notes
- 10% A (90-100)
- 20% A- (85-89)
- 25% B (80-84)
- 25% C (75-79)
- 20% D/E/F


## Gallery

### GoHugo website
- https://alexisz12.github.io/
- https://moonoxy.github.io/
- https://huangusr.github.io/
- https://lifelongcoding.github.io
- https://alexandreqiu.github.io/web/
- https://leo-fang-qaq.github.io/
- https://jialing78.github.io/
- https://hong-yue111.github.io/
- https://morganelu.github.io/

### reveal.js

### ML web app

### Final project


## Asking questions :question:

### Leveraging **[Gitee Issue](https://gitee.com/lundechen/cpp/issues)** for asking questions
By default, you should ask questions via **[Gitee Issue](https://gitee.com/lundechen/cpp/issues)**. Here is how:
- https://www.bilibili.com/video/BV1364y1h7sb/

### Principe
Here is the principle for asking questions:

>  **Google First, Peers Second, Profs Last.**

You are expected to ask questions via **[Gitee Issue](https://gitee.com/lundechen/cpp/issues)**. However, as a **secondary**  (and hence, less desirable, less encouraged) choice, you could also ask questions in the WeChat group.

> Why Gitee Issue? Because it's simply more **professional**, and better in every sense.

In Gitee Issue and the WeChat group, questions will be answered selectively. 

Questions won't be answered if:
- they could be solved on a simple Google search
- they are out of the scope of the course
- they are well in advance of the progress of the course
- professors think that it's not interesting for discussion

### Regarding personal WeChat chats:
- **Questions asked in personal WeChat chats will NOT be answered.**

Learning how to use Google & Baidu & Bing & ChatGTP to solve computer science problems is an important skill you should develop during this course.

For private questions, please send your questions by email to:
- lundechen@shu.edu.cn (Lunde Chen)

### Office visit

Office visit is NOT welcome unless you make an appointment at least one day in advance.

## Student Name List

| 学号/工号    | 姓名  |
| -------- | --- |
| 19124641 | 陆开昕 |
| 19124663 | 万远亮 |
| 19124715 | 张行行 |
| 20120127 | 李兆琪 |
| 20124695 | 邱奕博 |
| 20124711 | 袁嘉祾 |
| 20124725 | 张世博 |
| 20124727 | 翁留辰 |
| 20124738 | 宋鹏宇 |
| 20124757 | 王宇星 |
| 20124767 | 黄河  |
| 20124793 | 王雨杰 |
| 18124686 | 赵宇豪 |
| 18124689 | 闫炳坤 |
| 19124519 | 冯玥瑄 |
| 20124694 | 洪越  |
| 20124696 | 方鑫喆 |
| 20124726 | 马哲  |
| 20124733 | 杜若衡 |
| 20124769 | 李鑫宇 |
| 20124770 | 王泓杰 |
| 20124771 | 王楚涵 |
| 20124772 | 娄宇鑫 |
| 21124683 | 戴志成 |

## Online resources

1. 吴恩达机器学习系列：
    - https://www.bilibili.com/video/BV164411b7dx
1. 吴恩达深度学习系列：
    - https://www.bilibili.com/video/BV164411m79z

## F.A.Q

#### What characterizes this ML01 machine learning course?
- Stressful, fun and rewarding.

#### Do we have extra-course work?
-  Yes. A lot. 
- At least 10 hours of extra-course work each week is expected from you. 
    - 4 hours for course content & test preparation
    - 6 hours for your project (6 is the bare minimum, you might want to shoot up to 20 or 30 towards the end of the trimester).

#### What can I add as items to my CV after taking this course?
-  It's quite a lot. For example, AWS Amplify, AWS SageMaker, GitHub Pull Request, GitHub workflow, GoHugo, reveal.js, Cloud Computing, streamlit, fastapi, swagger, Docker, nginx, GitHub WebHook, machine learning, deep learning, next.js, MySQL, GitHub Actions, numpy, pandas, matplotlib, seaborn, plotly, sklearn, tensorflow, DQN, javascript, etc.

#### Why this course seems a bit different?
- Well, the prof draws inspirations from courses of Stanford, Berkeley and MIT.
    - http://cs231n.stanford.edu (Stanford)
    - https://c.d2l.ai/berkeley-stat-157 (Berkeley)
    - http://introtodeeplearning.com (MIT)
    - http://cs229.stanford.edu （Stanford）