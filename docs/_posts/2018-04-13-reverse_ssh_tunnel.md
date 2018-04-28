---
layout: post
title:  "Reverse SSH Tunnel"
author: huhuta
categories: Study
comments: true
---

# Reverse SSH Tunenel
Reverse ssh tunnel에 대해서 살펴 보겠습니다.

ssh 접속을 하고 싶지만 해당 machine이 NAT Gateway 뒤에 가려져 있어 public ip가 없거나, 
Firewall 때문에 접속이 불가능한 경우가 있습니다. 외부에서 내부로 접속 할 수 없는 경우입니다. 
remote host까지 가는 길을 모르거나 막혀있기 때문입니다.  
관점을 반대로 생각해서 내부에 존재하는 remote host에서 외부 서버에 들어올 수 있는 길을 
알려줄 수 있다면  어떨까요? (remote host가 인터넷이 되는 경우)
  

![gretel](http://www.byillust.com/qowhdfp/wp-content/uploads/sites/44737/2015/09/IMG_20150901_180728-e1441101686627.jpg)


## Example
아래가 명령어 예시 입니다. 

```
ssh -f -N -T -R22222:localhost:22 yourpublichost.example.com
```

터널을 만들고 -Remote entry point를 만듭니다.  
client 22222 port로 연결된 것들은 "localhost port 22"로 연결 됩니다.   
이제 remote 에서 ssh -p 22222를 할 경우 접속이 가능합니다. 

![image](https://hobosource.files.wordpress.com/2016/06/400px-ssh-reverse-connect.png)

지금은 22 번 port를 연결해놨지만 만약에 http 80 port를 reverse로 커넥션을 맺을 경우
web browser로 서버 접속도 가능 합니다.


```
ssh -f -N -T -R10080:localhost:80 yourpublichost.example.com
```
접속하려는 내부 서버에서 relay 서버 쪽에 reverse 터널을 열고


```
ssh -L 10080:localhost:10080 yourpublichost.example.com
```
내 local 머신의 10080 포트를 relay server10080 포트로 포워딩을 시킨다면 웹브라우저를 통해
통해 내부 서버에 접속할 수 있습니다.

## 주의할 점

회사 같은 보안이 중요한 곳에서는 reverse ssh tunnel 사용을 주의해야 합니다.  
회사 내부 정책에 위배되는 행위 일 수 있기 때문입니다. 내부망에 외부 네트워크 접속이 가능할 경우    
외부 공격에 노출 될 수 있습니다.



