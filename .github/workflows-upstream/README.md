# 옮겨 둔 업스트림 워크플로

이 포크(Munsik-Park/crewAI)는 codeinside 분석 시험용이다. 업스트림(crewAIInc/crewAI) 워크플로 19개를 `.github/workflows/`에서 이 폴더로 옮겼다. 이 폴더는 GitHub Actions가 읽지 않으므로 여기 있는 파일은 실행되지 않는다.

- 옮긴 시점: 포크 기준 커밋 `71d68117e38f620dff02ec1dc358f37a1f74a740` 바로 다음 커밋. Actions를 켜기 전에 옮겼다.
- 이 포크의 main은 업스트림과 동기화하지 않는다. 동기화하면 업스트림의 새 워크플로가 `.github/workflows/`로 다시 들어온다.
- 분석 사례로 쓸 업스트림 커밋은 SHA로 가져온다.
