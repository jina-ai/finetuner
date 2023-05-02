# {octicon}`light-bulb` What is Finetuner?

![Fine tuning](../imgs/finetuning-adjusting-dials.png)

The latest generation of large AI models does many very impressive things.
Anyone with access to the internet can try out talking with ChatGPT or making pictures with DALL-E or MidJourney. 
However, so far, very few businesses have been built around them.
This technology is very new, and it may take some time to see real productivity growth from them.
In the meantime, many existing businesses are looking at places in their own operations where they might usefully deploy AI.

Large AI models have been trained on large databases of unspecialized data, often texts and images taken from the internet with minimal filtering.
This gives them impressive-seeming abilities over a potentially unbounded array of scenarios.
However, when applied to specific problems, their performance can be disappointing.
Whatever your application is, most of the functionality of a large AI model is useless to you, while the functionality that is useful to you is not directly designed to address your use case.

Building a whole new AI model on the scale of recent breakthroughs, focused on just your business interests, is not very practical.
The major AI companies don't usually disclose how much time, energy, or money it takes to train their state-of-the-art models.
Estimates start in the millions of euros, with [some claiming GPT-4 cost over 100 million US dollars to train](https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/).
Even for the best-funded businesses, it is a challenge to find and retain sufficient engineering talent with experience in AI and another challenge to obtain the vast quantities of data required to train these models.
Successfully training very large models from scratch may take months on even the largest and fastest clusters at a cost of potentially millions of euros.
And this is before even addressing error handling and integration costs.

There is another way.

## Pre-training

One of the breakthroughs that led to today's impressive new AI models is the discovery that it often requires much less time and data to adapt an already trained model to a new task than to train a new one from scratch.
Techniques for “pre-training” networks — training them to do tasks that are indirectly related to your use case but for which there is ample data and easy-to-assess goals — are a big part of how we can create these large models.
For example, large language models are typically pre-trained using large quantities of text, with the training goal of filling in the blanks from the surrounding context.
For example:

> *On Saturday, New Jersey Gov. Phil Murphy _________ that Sept. 23 will _______ be declared "Bruce Springsteen Day" in ______ of the singer's birthday.*
>

By learning to do this task — one with little value by itself — AI models learn a great deal about the relationships between words.
Then, after the model has learned as much as it can from this task, developers train it to do more sophisticated things, like answering questions or holding conversations.

Image-processing AI models are pre-trained using similar techniques.
Typically, this means training to fill in the blank squares in pictures or by distorting or adding noise to images and training the model to “fix” them.
This teaches the model a lot about what kinds of things appear in pictures and what they should look like.

[//]: # (<style>)

[//]: # (    table: {border: none;})

[//]: # (</style>)

[//]: # (| |                                                               |)

[//]: # (|---|---------------------------------------------------------------|)

[//]: # (|![The Mona Lisa with a section cut out.]&#40;../imgs/MonaLisa1.png&#41;| ![The Mona Lisa with added blurring.]&#40;../imgs/MonaLisa2.png&#41;  |)


<table style="border: none;">
    <tr>
        <td width="45%">
            <figure>
                <img src="../_images/MonaLisa1.png" alt="The Mona Lisa with a section cut out."/>
                <figcaption style="text-align:center">The Mona Lisa with a section cut out.</figcaption>
            </figure>
        </td>
        <td width="45%">
            <figure>
                <img src="../_images/MonaLisa2.png" alt="The Mona Lisa with added blurring."/>
                <figcaption style="text-align:center">The Mona Lisa with added blurring.</figcaption>
            </figure>
        </td>
    </tr>
</table>

AI models that have already learned some relevant things are much easier to train than ones that start without knowing anything.
This is called *transfer learning*, and it's a very intuitive idea.
It's much easier to teach someone who knows how to drive a car to safely drive a large truck than to teach someone who has never driven at all.
AI models work in a similar way.

## Fine-tuning

You can take advantage of transfer learning to adapt AI to your business and your specific use cases without making a multi-year, multi-million euro investment.
The impressive performance of large AI models on general tasks means that they need much less data and training time to learn to do well at some specific task.

For example, let's consider an online fashion retailer that offers shoppers a search function.
There are AI models that match pictures to descriptions, but they are trained on a very wide variety of images.
It does the retailer no good to have AI that can efficiently identify pictures of dogs and cats when they want an AI that can tell the difference between jeans and dress slacks or between an A-line skirt and a pleated skirt.

An AI model that can recognize thousands of different objects has already learned how to make fine distinctions based on the features of images.
Training it to know the difference between chinos and cargo pants takes relatively few examples compared to training an AI model that doesn't even know what pants are!

This kind of training is called *fine-tuning* because most of what it does is not really learning new things but learning to focus the things that the model already knows on the specific tasks you have in mind.

Although performance is almost always going to be better the more training data you have, we have found that [mere hundreds of items of training data are enough to get most of the gains](https://jina.ai/news/fine-tuning-with-low-budget-and-high-expectations/) from fine-tuning in many highly realistic test cases.
Acquiring a few hundred examples is a much cheaper proposition than acquiring the millions to billions of data items used to train the original model.

This makes fine-tuning an extremely attractive value proposition for anyone looking to integrate AI into their business.

## Finetuner from Jina AI

Jina AI's Finetuner is a flexible and efficient cloud-native solution for fine-tuning state-of-the-art AI models.
It provides an intuitive Python interface that can securely upload your data to our cloud and return a fine-tuned model without requiring you to invest in any special hardware or tech stack.
Our solution includes testing and evaluation methods that quantify performance improvements and provide intuitive reports, so you can see directly how much value fine-tuning adds.

It is impossible for us — or anyone else — to guarantee a specific outcome from a machine learning process.
However, we are unaware of any case in which fine-tuning with properly representative data did not improve the task-specific performance of an AI model.

We strive to keep the Finetuner as intuitive and easy to use as possible, but we understand that machine learning can be an inherently messy process involving choices that no software can ever automate.
For this reason, we offer [Finetuner+](https://finetunerplus.jina.ai/), a service where our engineers collaborate with you in planning and executing AI integration.
We can:

- Advise you on selecting an AI model appropriate to your use case.
- Help you to acquire and prepare training data for the Finetuner.
- Manage and evaluate Finetuner runs for you.
- Assist you in integrating the resulting AI models into your business processes.

Furthermore, if your data security needs are very strict, we can assist you in setting up and running Finetuner on-site at your company.

If you are looking at integrating AI into your business, fine-tuning is essential to getting the most value out of your AI investments.
Jina AI can help with software, services, and consultants for organizations of all kinds, sizes, and technical experience.