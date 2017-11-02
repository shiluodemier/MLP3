import numpy 
import nibabel 
import pp
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn import neighbors

def hog3d(img):
    (nx,ny,nz)=img.shape
    Gx,Gy,Gz=numpy.gradient(img)
    hgram=numpy.zeros((12,12),dtype='float32')
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                agx=Gx[i,j,k]
                agy=Gy[i,j,k]
                agz=Gz[i,j,k]
                if agx==0.0 and agy==0.0 and agz==0.0: continue
                if agx==0.0 and agy==0.0: agx+=0.001
                g=numpy.array([agx,agy,agz])
                glen=numpy.sqrt(g.dot(g))
                g0=numpy.array([1,0,0])
                g0len=numpy.sqrt(g0.dot(g0))
                g1=numpy.array([agx,agy,0])
                g1len=numpy.sqrt(g1.dot(g1))
                angle1=numpy.arccos(round(g1.dot(g0)/(g0len*g1len),5))
                if agy<0: angle1=2.0*numpy.pi-angle1
                angle2=numpy.arccos(round(g.dot(g1)/(glen*g1len),5))
                if agz<0: angle2=2.0*numpy.pi-angle2
                loc1=int(angle1/(numpy.pi/6.0))%12
                loc2=int(angle2/(numpy.pi/6.0))%12
                hgram[loc1,loc2]+=glen
    hgram=numpy.reshape(hgram,(12*12))
    return hgram

def imgprocess(numfrom,numto,setkind,dividerate=4,sizer=176,sizeq=208,sizep=176):
    traindata=numpy.empty((numto-numfrom+1,dividerate*dividerate*dividerate*144),dtype='float32')
    for i in range(numfrom,numto+1):
        print 'Loading '+setkind+'_'+str(i+1)+'.nii'
        img=nibabel.load('data/set_'+setkind+'/'+setkind+'_'+str(i+1)+'.nii')
        data_array=img.get_data()
        imgfeature=numpy.array([],dtype='float32')
        for p in range(0,sizep,sizep/dividerate):
            for q in range(0,sizeq,sizeq/dividerate):
                for r in range(0,sizer,sizer/dividerate):
                    pitch=data_array[p:p+sizep/dividerate,q:q+sizeq/dividerate,r:r+sizer/dividerate,0]
                    imgfeature=numpy.concatenate((imgfeature,hog3d(pitch)))
        traindata[i-numfrom,:]=imgfeature
    return traindata

def loadimage(num,setkind,batches=4,dividerate=4):
    traindata=numpy.array([],dtype='float32')
    remained=num%(batches)
    if remained!=0: batches-=1
    numperbatch=num//(batches)
    inputs=()
    for i in range(batches):
        inputs=inputs+((i*numperbatch,(i+1)*numperbatch-1,setkind),)
    if remained!=0:
        inputs=inputs+((batches*numperbatch,num-1,setkind),)
    print inputs
    ppservers = ()
    job_server = pp.Server(6, ppservers=ppservers)
    jobs = [(input, job_server.submit(imgprocess,input,(hog3d,),("numpy","nibabel"))) for input in inputs]
    counter=0
    for input,job in jobs:
        if counter==0:traindata=job()
        else : traindata=numpy.concatenate((traindata,job()))
        counter+=1
        print 'Finish'+str(counter)
    return traindata
def loadtarget(num):
    file_object = open('targets.csv')
    ageset=numpy.arange(num)
    counter=0
    for tline in file_object:
        if tline!='':
            ageset[counter]=int(tline)
            counter+=1
    file_object.close()
    return ageset
def saveimageinfo(array_data,setkind):
    f = open(setkind+'data.csv', 'w')
    d1,d2=array_data.shape
    f.write(str(d1)+','+str(d2)+"\n")
    for i in range(d1):
        str0=','.join(str(j) for j in array_data[i,:])
        str0=str0+"\n"
        f.write(str0)
    f.close()
   
def loadcsv(setkind):
    f = open(setkind+'data.csv') 
    line =f.readline().split(',')
    num= int(line[0])
    size=int(line[1])
    data_array=numpy.empty((num,size),dtype='float32')
    for i in range(num):
        line =f.readline().split(',')
        data_array[i,:]=[float(j) for j in line]
    f.close()
    return data_array
        
if __name__=="__main__":
    '''
    traindata=loadimage(278,"train",12)  
    testdata=loadimage(138,"test",12)
    saveimageinfo(traindata, 'train')
    saveimageinfo(testdata, 'test') 
    ''' 
    traindata=loadcsv("train")  
    testdata=loadcsv("test") 
    print traindata.shape
    print testdata.shape
    targets=loadtarget(278)
    print 'Done image load'
    '''
    pca = PCA(n_components=278)
    pca.fit(traindata)
    traindata=pca.transform(traindata)
    testdata=pca.transform(testdata)
    print 'Done PCA'
    '''
    skb=SelectKBest(f_classif, k=200)
    skb.fit(traindata, targets)
    traindata=skb.transform(traindata)
    testdata=skb.transform(testdata)
    print "Done selection"
    clf = neighbors.KNeighborsClassifier(5, weights='distance')
    clf.fit(traindata,targets)
    clf.predict(testdata)
    results=clf.predict_proba(testdata)
    print results
    print 'Done Classification'
    output=open('sampleSubmission.csv','w')
    output.write("ID,Prediction\n");
    for i in range(138):
        prob=str(i+1)+','+str(results[i,1])+"\n"
        output.write(prob)
    output.close()
    
                       
                
                