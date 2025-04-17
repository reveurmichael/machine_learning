# YouTube Subtitle Enhancement and Translation Project

Those video srt files are downloaded from YouTube.

In fact, you can see the id of those videos in their file names, e.g. for `2015 NBA Finals Mini-Movie, Games 1-6 + More [3pxKO_-Cp-o].srt`, the id is `3pxKO_-Cp-o`. Hence, its corresponding video is `https://www.youtube.com/watch?v=3pxKO_-Cp-o`.

You can download the srt files as well as the video files with the tool `yt-dlp`, or `you-get`.

## Current Challenges

Problems that I have identified:
- Some of the srt in English might not be good enough. Though most should be fine. One perticular keyword, for example, `foreign`, can be very misleading.
- If we translate the srt files into Chinese with non-LLM AI tools, the result might not be good, sometimes very bad. See [practice-2-output-srt-files](../week-4/practice-2-output-srt-files/)

## Project Objectives

I would like you guys to:
- identify things that can be pitfalls in the srt files (e.g. the keyword `foreign`).
- explore different tools to improve the quality of the srt files. Maybe youtube srt files could be enhanced by Whisper, it will be like information fusion from two sources.
- explore different LLM tools to translate the srt files into Chinese. 
- identify the minimum viable LLM models (e.g. Qwen2.5:3B) to translate the srt files for the videos.
- how to make the translation result better (e.g. prompt engineering, context aware, continuity checks to ensure logical sentence flow across subtitle segments).
- how much time will it take to translate the srt files for all videos (let's say 1000 srt files).

However, as long as your srt files (En plus translated Chinese) are good enough, you can skip the above objectives. 


## [Optional] Human-in-the-Loop Validation
   - Implement targeted human review for segments flagged as potentially problematic
   - Create an efficient interface for reviewers to quickly approve or correct translations
   - Develop a feedback mechanism to improve system performance based on corrections



