# T-bot :  RAG for csv files

https://github.com/yoojchul/csv2mysql 는 csv 파일을 MySQL로 변환하는 프로그램인데 이를 RAG에 적용한 사례입니다. 프로젝트 이름을 T-bot이라 칭하고 코드 변환을 위하여 아래 프롬프트로 Claude AI를 사용했습니다 

```
csv 파일을 외부 리소스로 하는 RAG 시스템을 openwebui를 이용하여 구성한다.  유저는 다수의 csv 파일을 업로드하고 RAG 시스템은
csv 파일 이름과 header 정보를 milvus에 저장하고 파일 자체는 mysql에 저장한다. 


쿠키를 이용하여 milvus의 collection, mysql database 이름을 기억하고 화면 상단에 표시하라. collection 혹은 database가 없으면
"No data" 표시하라. milvus collection과 mysql database 이름은 동일하다. 이하 database라고 지칭한다. 한 user당 하나의
database를 허용한다. 

database 이름 옆에 삭제 버튼 필요하고 삭제 버튼이 클릭되면 쿠키에 기록된 database,  해당 milvus의 collection
그리고 mysql database 를 삭제한다. 

openwebui를 통해 업로드를 지원하는데 한번에  여러 파일을 업로드할 수 있도록 허용하라.  파일 갯수는 제한이 없고 한번에
올릴 수 있는 파일의 총량은 1MB로 제한한다.  업로드 파일 zip 형식이면 자동적으로 압축을 풀어서 업로드하라. 업로드 파일은
/var/lib/mysql-files에 폴더를 만들고 이동시켜라. 폴더 이름은 milvus의 collection과 mysql의 database 이름과 동일하게 하라.
아래부터는 database로 지칭한다.   database 이름은 사용자가 지정하게 한다. 

이미 쿠키에 기록된 collection과 database가 있을 때 업로드를 시도하면 "존재하는 data가 있어 삭제가 필요하다"는 메세지를 보내라

업로드한 파일들에 대해 vecotor DB를 만들때는  csv2recap.py을,  mysql로 변환할 때는 csv2mysql.py를 수정해서 사용하라.
csv2recap.py를 실행할 동안에는 모래시계 그림을 화면에 표시하라.  csv2mysql.py를 실행하는 중 한개의 파일 작업이 끝날 때마다
파일 헤더 포함 상위 5줄만 화면에  테이블 형식으로  순차적으로 그려라.  column이 많아 한 화면에 나오지 못하면  수평으로
scroll 할 수 있도록 만들어라.

질문을 받아 답변을 실행하는 프로그램은 search.py 을 수정해서 사용하라. 
```

모든 화면 출력 내용은 <database이름>.log 파일 이름으로 /var/lib/mysql-files에 기록한다. 

위 내용을 만족하도록 프로그램을 구성하라.

변경및 추가된 프로그램와 다음과 같다. 

```
  T-bot/
  ├── app.py          ← FastAPI 백엔드 (신규)
  ├── csv2recap.py    ← callback 파라미터 추가
  ├── csv2mysql.py    ← callback 추가 + SET 버그 수정
  ├── search.py       ← run_query() 함수 추가
  ├── static/
  │   └── index.html  ← 완성형 SPA 프론트엔드 (신규)
  └── run.sh          ← 실행 스크립트 (신규)
```

 http://raison.hokepos.net:8000 에 접속해서 테스트해볼 수 있다. 
