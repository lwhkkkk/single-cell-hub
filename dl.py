import urllib.request, ssl, os
items={
'scGPT':'https://arxiv.org/pdf/2305.16175',
'Geneformer':'https://www.nature.com/articles/s41586-023-06139-9.pdf',
'scFoundation':'https://www.nature.com/articles/s41592-024-02305-7.pdf',
'scBERT':'https://liuzi919.github.io/papers/PLM/scBERT_Nature%20Machine%20Intelligence_2022.pdf'}
c=ssl._create_unverified_context()
for n,u in items.items():
 p=f'single_cell_models/{n}/paper/{n}.pdf'
 try:
  req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
  x=urllib.request.urlopen(req,context=c,timeout=120).read()
  open(p,'wb').write(x); print(n,len(x))
 except Exception as e: print(n,'FAIL',e)
