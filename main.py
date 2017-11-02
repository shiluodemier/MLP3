import nibabel 
import cv2
import numpy
import pp
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.svm import SVC
import scipy.stats
def cut(data,n,mix=0): #First cut to divide imge in regular samples.
    dx,dy,dz=data.shape
    cuts=numpy.zeros((1,n,n,n),dtype="float32")
    counter=0
    for i in range(0,dx,n-mix):
        if(i+n>dx): continue
        for j in range(0,dy,n-mix):
            if(j+n>dy): continue
            for k in range(0,dz,n-mix):
                if(k+n>dz): continue
                if(i+k+j==0):
                    cuts[0,:,:,:]=data[0:n,0:n,0:n]
                    continue
                newcut=numpy.empty((1,n,n,n),dtype="float32")
                newcut[0,:,:,:]=data[i:i+n,j:j+n,k:k+n]
                cuts=numpy.concatenate((cuts,newcut),axis=0)
                counter+=1
    print counter
    return cuts
def cut2(data,nx,ny,nz): #Second cut to make sure samples in nromal size.
    dx,dy,dz=data.shape
    sx=dx//nx+1
    sy=dy//ny+1
    sz=dz//nz+1
    cuts=numpy.zeros((nx*ny*nz,sx,sy,sz),dtype="float32")
    for i in range(0,dx,sx):
        if(i+sx>dx):
            lx=dx-sx
            hx=dx
        else:
            lx=i
            hx=i+sx
        for j in range(0,dy,sy):
            if(j+sy>dy):
                ly=dy-sy
                hy=dy
            else:
                ly=j
                hy=j+sy
            for k in range(0,dz,sz):
                if(k+sz>dz):
                    lz=dz-sz
                    hz=dz
                else:
                    lz=k
                    hz=k+sz
                xx=i/sx
                yy=j/sy
                zz=k/sz
                cuts[xx*ny*nz+yy*nz+zz,:,:,:]=data[lx:hx,ly:hy,lz:hz]
    return cuts
def cut3(data): #Preprocess
    cimg=numpy.array(data[18:155,18:190,10:150,0],dtype="float32")
    return cimg
    
def compare(): #Compare healthy samples and unhealthy ones
    ageset=numpy.arange(278)
    file_object = open('targets.csv')
    counter=0
    for tline in file_object :
        if tline!='':
            ageset[counter]=int(tline)
            counter+=1
    file_object.close()
    healthycounter=0
    diseasecounter=0
    healthyarray=numpy.empty((8,176,208,176),dtype="float32")
    diseasearray=numpy.empty((8,176,208,176),dtype="float32")
    for i in range(278):
        img=nibabel.load('data/set_train/train_'+str(i+1)+'.nii')
        data_array=img.get_data()
        if (healthycounter<8 and ageset[i]==1) :
            healthyarray[healthycounter,:,:,:]=data_array[:,:,:,0]
            healthycounter+=1
            print i
        if (diseasecounter<8 and ageset[i]==0):
            diseasearray[diseasecounter,:,:,:]=data_array[:,:,:,0]
            diseasecounter+=1
            print i
        if (diseasecounter==3 and healthycounter==3): break
    maxv1=numpy.amax(diseasearray[:,:,:,88])
    maxv2=numpy.amax(healthyarray[:,:,:,88])
    maxv=max(maxv1,maxv2)
    imgunhealthy=diseasearray[0,:,:,88]/maxv
    for i in range(1,8):
        imgunhealthy=numpy.concatenate((diseasearray[i,:,:,88]/maxv,imgunhealthy),axis=1)
    imghealthy=healthyarray[0,:,:,88]/maxv
    for i in range(1,8):
        imghealthy=numpy.concatenate((healthyarray[i,:,:,88]/maxv,imghealthy),axis=1)
    p=numpy.concatenate((imghealthy,imgunhealthy),axis=0)
    cv2.imshow("compare",p)
    cv2.waitKey (0)  

def hsg(data,n=60,maxv=3000.0):
    ncube,dx,dy,dz=data.shape
    hsgarray=numpy.zeros((ncube,n),dtype="uint16")
    ni=maxv/n
    for i in range(ncube):
        for x in range(dx):
            for y in range(dy):
                for z in range(dz):
                    index=int(data[i,x,y,z]//ni)
                    if (index>n-1): index=n-1
                    hsgarray[i,index]+=1
    return hsgarray

def multip(dfrom,dto,dkind,nx=9,ny=11,nz=9):
    h1=nx*ny*nz
    h2=60
    returnarray=numpy.empty((dto-dfrom+1,h1,h2),dtype="uint16")
    for i in range(dfrom,dto+1):
        img=nibabel.load('data/set_'+dkind+'/'+dkind+'_'+str(i+1)+'.nii')
        data_array=cut3(img.get_data())
        cuts=cut2(data_array,nx,ny,nz)
        hsgarray=hsg(cuts)
        returnarray[i-dfrom,:,:]=hsgarray
    return returnarray

def preprocess(num,dkind,batches=4,nx=9,ny=11,nz=9):
    remained=num%(batches)
    if remained!=0: batches-=1
    numperbatch=num//(batches)
    redata=numpy.zeros((numperbatch,nx*ny*nz,60),dtype="uint16")
    print redata.shape
    inputs=()
    for i in range(batches):
        inputs=inputs+((i*numperbatch,(i+1)*numperbatch-1,dkind),)
    if remained!=0:
        inputs=inputs+((batches*numperbatch,num-1,dkind),)
    print inputs
    ppservers = ()
    job_server = pp.Server(6, ppservers=ppservers)
    jobs = [(input, job_server.submit(multip,input,(cut2,cut3,hsg),("numpy","nibabel"))) for input in inputs]
    counter=0
    for input,job in jobs:
        if counter==0: redata=job()
        else : redata=numpy.concatenate((redata,job()),axis=0)
        counter+=1
        print 'Finish'+str(counter)
    return redata
def saveimageinfo(array_data,setkind):
    f = open(setkind+'datahsg.csv', 'w')
    d1,d2=array_data.shape
    f.write(str(d1)+','+str(d2)+"\n")
    for i in range(d1):
        str0=','.join(str(j) for j in array_data[i,:])
        str0=str0+"\n"
        f.write(str0)
    f.close()
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
def loadcsv(setkind):
    f = open(setkind+'datahsg.csv') 
    line =f.readline().split(',')
    num= int(line[0])
    size=int(line[1])
    data_array=numpy.empty((num,size),dtype='float32')
    for i in range(num):
        line =f.readline().split(',')
        data_array[i,:]=[float(j) for j in line]
    f.close()
    return data_array

def ttest(cfrom,cto,data0,data1):
    tresult=numpy.zeros((cto-cfrom+1),dtype="float64")
    for i in range(cfrom,cto):
        a0=data0[:,i,:]
        a1=data1[:,i,:]
        
    
def ttestprocess():
    
    file_object = open('targets.csv')
    ageset=numpy.arange(278)
    counter0=0
    counter1=0
    counter=0
    for tline in file_object:
        if tline!='':
            ageset[counter]=int(tline)
            counter+=1
            if ageset[counter]==0 : counter0+=1
            else: counter1+=1
    file_object.close()
    
    nx=9
    ny=11
    nz=9
    img=nibabel.load('data/set_train/train_1.nii')
    data_array=cut3(img.get_data())
    cuts=cut2(data_array,nx,ny,nz)
    nc,sx,sy,sz=cuts.shape
    cuts=numpy.reshape(cuts,(nc,sx*sy*sz))
    traincubes0=numpy.zeros((counter0,nc,sx*sy*sz))
    traincubes1=numpy.zeros((counter1,nc,sx*sy*sz))
    counter0=0
    counter1=0
    for i in range(278):
        img=nibabel.load('data/set_train/train_'+str(i+1)+'.nii')
        data_array=cut3(img.get_data()) 
        cuts=cut2(data_array,nx,ny,nz) 
        cuts=numpy.reshape(cuts,(nc,sx*sy*sz))
        if ageset[i]==0:
            traincubes0[counter0,:,:]=cuts
            counter0+=1
        else:
            traincubes1[counter1,:,:]=cuts
            counter1+=1
    
        
if __name__=="__main__":
    compare()

    
    
    
    
    
    
        