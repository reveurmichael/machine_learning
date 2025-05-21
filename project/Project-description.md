# YouTube Subtitle Enhancement and Translation Project

In the folder [srt-files](./srt-files), you can find a bunch of video srt files downloaded from YouTube.

In fact, you can see the id of those videos in their file names, e.g. for `2015 NBA Finals Mini-Movie, Games 1-6 + More [3pxKO_-Cp-o].srt`, the id is `3pxKO_-Cp-o`. Hence, its corresponding video is `https://www.youtube.com/watch?v=3pxKO_-Cp-o`.

You can download the srt files as well as the video files with the tool `yt-dlp`, or `you-get`. Or, simply Google "download youtube videos online".

## Current Challenges

Problems that I have identified:
- Some of the srt in English might not be good enough. Though most should be fine. One perticular keyword, for example, `foreign`, can be very misleading. Or, sometimes (maybe very often?), the original srt file downloaded from YouTube might not be good enough.
- If we translate the srt files into Chinese/French with non-LLM AI tools (e.g. `Helsinki-NLP/opus-mt-en-zh`), the result might not be good, sometimes very bad. See [week-4/practice-2-output-srt-files](../week-4/practice-2-output-srt-files/). From my first experiences of trial and error, there is no reason to expect better results with other non-LLM tools.

## Project Objectives

I would like you guys to:
- identify things that can be problematic/pitfalls in the downloaded srt files (e.g. the keyword `foreign`).
- explore different methods to improve the quality of the downloaded srt files. For example, maybe youtube srt files could be enhanced by Whisper (local model, as shown in [week-4/practice-2-video-transcription-with-ai-Helsinki-NLP-opus-mt-en-zh.md](../week-4/practice-2-video-transcription-with-ai-Helsinki-NLP-opus-mt-en-zh.md)), it will be like information fusion from two sources (whisper transcript and original srt file combined together, to help mitigate the timestamps issue and other issues). This might be too CPU-intensive and time-consuming, hence it can be just a bit of simple exploration; if it proves effective, you can go deeper.
- **it's important to emphasize that having a high quality EN srt file is 50% of the objective of the work.**
- explore different LLM models to translate the downloaded srt files into Chinese/French (maybe Local (preferred), or, if local is not enough, Cloud LLM API). 
- identify the minimum viable LLM models (e.g. `Qwen3`) to translate the srt files for the videos with acceptable quality.
- how to make the translation result better (e.g. prompt engineering, context awareness, continuity checks to ensure logical sentence flow across subtitle segments, feed into the information of the video (video title, description, channel title, channel description, etc.), etc.).
- how much time will it take to translate the srt files for all videos (let's say 1000 srt files). Propose an efficient processing pipeline to minimize/balance/optimize time and resource consumption. Therefore, your code need to be generalizable and scalable.

However, as long as your subtitle files (both in English and the translated Chinese/French) are of sufficient quality, you can safely choose to skip the above mentioned objectives.

And yes, your project should be implemented in Python code.

The project can be quite open, it can be research oriented, or application oriented, or a blend of both, depending on your taste and styling. I have no idea what will come out from this project yet. Hence, just play hard, work harder, have fun and surprise me.


It can be as simple as:
- Python read EN srt file line by line, use LLM to improve EN content
- Python read EN srt file line by line, use LLM to translate it into CN

But there are (hidden) pitfalls everywhere. So get your hands dirty is the most important thing.


## [Optional] what information can be get from yt-dlp --dump-json ?

| Field                | Description                                                              |
| -------------------- | ------------------------------------------------------------------------ |
| `id`                 | Video ID                                                                 |
| `title`              | Video title                                                              |
| `description`        | Full video description                                                   |
| `uploader`           | Channel name                                                             |
| `uploader_id`        | Channel ID                                                               |
| `upload_date`        | Date in `YYYYMMDD` format                                                |
| `duration`           | Length in seconds                                                        |
| `view_count`         | Total views                                                              |
| `like_count`         | Likes (if available)                                                     |
| `formats`            | List of available video/audio formats (codec, resolution, bitrate, etc.) |
| `thumbnails`         | List of thumbnail URLs and resolutions                                   |
| `webpage_url`        | The original video URL                                                   |
| `categories`         | List of video categories                                                 |
| `tags`               | List of tags assigned to the video                                       |
| `channel_url`        | Channel homepage URL                                                     |
| `subtitles`          | Dictionary of available subtitles (if any)                               |
| `automatic_captions` | Auto-generated captions info (if any)                                    |
| `is_live`            | Whether it’s/was a livestream                                            |
| `chapters`           | List of chapter start/end times (if defined)                             |


## [Optional] Human-in-the-Loop Validation with Streamlit
   - Implement targeted human review for segments flagged as potentially problematic
   - Create an efficient interface for reviewers to quickly approve or correct translations
   - Develop a feedback mechanism to improve system performance based on corrections


## Some random thoughts

- LLM can be involved in multiple ways, multiple stages, multiple parts of the process.
- LLM A might be good at fixing the timestamps/overlapping issue (with reasoning capabilities?)
- LLM B might be good at improve the EN srt
- LLM C might be good at translating EN to CN/FR
- LLM D might be good at summarizing the text. This summary might be feeded into LLM B for the latter's task.

You could check out the concept of MoE (mixture of experts) which is very a la mode.

Also, it's always good to double check your final results with

- video downloaded by yt-dlp
- VLC playing the video with srt file loaded

It's a very good idea to get mistralai/deepseek/hunyuan involved, at some point of the task, provided that you don't go bankrupt with the cost.


I am a research guy so those ideas are quite research-oriented.

You can stick with your styling of going for good product, or good enginnering,  or being visually convincing, or other stuffs.

Also, a well balanced workload can help everyone, just like in a company and in a research team.

## QA 1

Q: Hello sir, i wanted to know with my group, how many translated srt do you need and how ? Do you need the translated ones in a doc or not, like just examples in the video and let you try with the initial srt and our code solution 

A: Thank you for your question. If you solution is good and scalable, then you can translate as many srt files as possible (in the srt-files folder). After all, it's just python run the code. Things are quite open, for me the key is that I see that you are working (a lot), you are giving your insights and you are trying to bring up viable solutions. I won't check all your srt files, but I will check the video, the github/gitee repo (in that you are working as a team and your code make sense).

As always, take it easy, have fun, work a lot in an enjoyable way, don't try to make a perfect project, but a working one that you are proud of. 

It's AI era. Having fun is the most important thing.

A bit of fuzziness/guaussian noise is very welcome, just as in AutoEncoder and Diffusion Model :-)

