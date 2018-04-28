---
layout: post
title:  "Webstorm file wathcer 설정하기"
author: huhuta
categories: Study
comments: true
---

# Webstorm File Wathcer

Webstorm에는 file wathcer라는 기능이 있습니다.  
이 기능을 사용하면 파일이 수정되었을때 특정 동작을 일으킬 수 있습니다.  
  
Prettier를 사용 하여 파일이 저장될 때 reformat 시킨다면 아래와 같이 설정해주면 됩니다.   

아래와 같이 설정 해줍니다.

> File Type: JavaScript
> Scope: Project Files
> Program: full path to .bin/prettier or .bin\prettier.cmd in the project's node_module folder.  
> Arguments: --write [other options] $FilePathRelativeToProjectRoot$  
> Output paths to refresh: $FilePathRelativeToProjectRoot$  
> Working directory: $ProjectFileDir$  
> Auto-save edited files to trigger the watcher: Uncheck to reformat on Save only.  

---

![jetbrain](https://prettier.io/docs/assets/webstorm/file-watcher-prettier.png)
[출처 jetbrain 사이트](https://prettier.io/docs/en/webstorm.html)
