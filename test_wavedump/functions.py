import numpy as np
# from scipy import integrate
from scipy.signal import find_peaks
import qetpy as qp
import h5py
fs = 1/2e-9

def process_data(trace, recordlength):
    try:
        traceR = trace.reshape(int(np.floor(len(trace)/recordlength)),recordlength)
    except ValueError:
        trace = trace[:int(np.floor(len(trace)/recordlength))*recordlength,]
        traceR = trace.reshape(int(np.floor(len(trace)/recordlength)),recordlength)
    data= traceR-np.expand_dims(traceR[:,100:2000].mean(axis=1), axis=1)
#     print ('input trace length',trace.shape, 'processed trace shape',  data.shape)
    return data
def create_template(traces):
    template = np.mean(traces, axis = 0)
    template_norm= -template/min(template)
    return template,template_norm

def clippedwaveformsindex(data, clippingstart = -15100):
    clippedIdx = []
    data_max_idx = np.argmin(data, axis=1) 
    for i in range(len(data)):
        if data[i][data_max_idx[i]] < clippingstart:
            clippedIdx.append(i)
    return clippedIdx


def glitchesindex(data, threshold = 5):
    glitchesIdx = []
    
    std = ((data[:, 100:1000].std(axis=1)).reshape(len(data),1))
    t0idx = np.argmax(data<-(10*std), axis=1)   
        
    data_max_idx = np.argmin(data, axis=1)
    
    for i in range(len(data)):
        if data_max_idx[i] - t0idx[i] < threshold:
            glitchesIdx.append(i)
    return glitchesIdx


def pileupindex(data, ctype):
    pileupIdx, noisyIdx = [], []

    energy = np.sum(data/np.min(data, axis =1)[:, None], axis =1)
    
    if ctype == 'pure': distancefactor, prominencefactor = 15,  0.085
    else: distancefactor, prominencefactor = 200, 0.4
    if ctype == 'pure': threshold_min, threshold_max = 15,40
    else: threshold_min, threshold_max = 350,900
        
    data_max_idx = np.argmin(data, axis=1)
    data_max = data[np.arange(len(data_max_idx)),data_max_idx]  
    for i in range(0, len(data)):
        peaks, peak_properties = find_peaks(-data[i], prominence =(-data_max[i]*prominencefactor), distance=distancefactor)
        if len(peaks)>1 :
            pileupIdx.append(i)
        if energy[i] > threshold_max or energy[i] < threshold_min:
            noisyIdx.append(i)
    return pileupIdx, noisyIdx

def OF_calct0(signal, template_norm, psd_noise, fs, recordlength):
    amp, t0, chi2= [], [], []
    for trace in np.arange(len(signal)):
        OF = qp.OptimumFilter(signal[trace], template_norm, psd_noise, fs) # initialize the OptimumFilter class
        amp_withdelay, t0_withdelay, chi2_withdelay= OF.ofamp_withdelay() # t0_withdelay : time shift calculated for the pulse (in s).
        amp.append(amp_withdelay)
        t0.append(t0_withdelay)
        chi2.append(chi2_withdelay)
    return amp, t0, chi2



 
# energy in [ADC] / energyfactor = energy in [MeV]
'''energyfactor_pure 6243
energyfactor_Tl 118252

longtime_pure 250
shorttime_pure 24

longtime_Tl 11000
shorttime_Tl 600'''
def Ecalibration (amp, ctype):
    if ctype == 'Tl':
        return amp/118252
    if ctype == 'pure':
        return amp/6243
    
    
    
def hdf(File):
    print(h5py.File(File).keys())

class sort():
    def func(self, File, key, ctype, level = None):
        file = h5py.File(File)
        self.var = file[key][:]

        deltachi2 = file['Delta_chi2'][:]
        amppil_EM = file['amp_pileup_EM'][:]
        t0pil_EM = file['t0_pileup_EM'][:]
        amppil_Had = file['amp_pileup_Had'][:]
        t0pil_Had = file['t0_pileup_Had'][:]

        gli= np.where(file['glitch'][:]==True)[0]
        clip= np.where(file['clipped'][:]==True)[0]
        pileup= np.where(file['pileup'][:]==True)[0]
        noisy= np.where(file['noisypulse'][:]==True)[0]
        self.select = np.unique(np.concatenate((gli,clip,pileup,noisy),0))
        
        self.var_PS_removed = np.take(file[key][:], self.select)
        self.var_PS = np.delete(file[key][:], self.select)  

        if ctype == 'Tl' :
            select2 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and  (abs(amppil_EM[i] )>1000 or abs(t0pil_EM[i]*fs )>2000)]
            select3 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and (abs(amppil_Had[i] )>1000 or abs(t0pil_Had[i]*fs )>2000)]
            select6 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and abs(amppil_EM[i] )>300 and (t0pil_EM[i]*fs )<-50]
            select7 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and abs(amppil_Had[i] )>300 and (t0pil_Had[i]*fs )<-50]
            self.select_FT = list(set(list(self.select)+ select3+select2+ select6+select7))

        if ctype == 'pure':
            select2 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and  (abs(amppil_EM[i]) > 200 or  abs(t0pil_EM[i]*fs) > 200)]
            select3 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and (abs(amppil_Had[i] ) > 200 or abs(t0pil_Had[i]*fs) > 200)]
            self.select_FT = list(set(list(self.select)+ select3+select2))

        self.var_PS_FT_removed = np.take(file[key][:], self.select_FT)
        self.var_PS_FT = np.delete(file[key][:], self.select_FT)  
        
        if level == 'preselected_removed':
            return self.var_PS_removed
        if level == 'preselected':
            return self.var_PS
        if level == 'preselected_FT':
            return self.var_PS_FT
        if level == 'preselected_FT_removed':
            return self.var_PS_FT_removed
        
        
        
     
       
class variables():
    def var(self, File, ctype, examine):
        file = h5py.File(File)

        deltachi2 = file['Delta_chi2'][:]
        chi2_EM = file['chi2_withdelay_EM'][:]
        amppil_EM = file['amp_pileup_EM'][:]
        self.t0pil_EM = file['t0_pileup_EM'][:]
        chi2pil_EM = file['chi2_pileup_EM'][:]
        chi2_Had = file['chi2_withdelay_Had'][:]
        amppil_Had = file['amp_pileup_Had'][:]
        self.t0pil_Had = file['t0_pileup_Had'][:]
        chi2pil_Had = file['chi2_pileup_Had'][:]
        self.deltachi2 = file ['Delta_chi2'][:]

        gli= np.where(file['glitch'][:]==True)[0]
        clip= np.where(file['clipped'][:]==True)[0]
        pileup= np.where(file['pileup'][:]==True)[0]
        noisy= np.where(file['noisypulse'][:]==True)[0]
        self.select = np.unique(np.concatenate((gli,clip,pileup,noisy),0))
        if examine: 
            self.amp_EM_PS_removed = np.take(file ['amp_withdelay_EM'][:],self.select)
            self.t0_EM_PS_removed = np.take(file ['t0_withdelay_EM'][:],self.select)
        #     self.chi2_EM_PS_removed = np.take(file ['chi2_withdelay_EM'][:],self.select)
            self.amppil_EM_PS_removed = np.take(file['amp_pileup_EM'][:], self.select)
            self.t0pil_EM_PS_removed = np.take(file['t0_pileup_EM'][:], self.select)
        #     self.chi2pil_EM_PS_removed = np.take(file['chi2_pileup_EM'][:], self.select)    
            self.amp_Had_PS_removed = np.take(file ['amp_withdelay_Had'][:],self.select)
            self.t0_Had_PS_removed = np.take(file ['t0_withdelay_Had'][:],self.select)
            self.chi2_Had_PS_removed = np.take(file ['chi2_withdelay_Had'][:],self.select)
            self.amppil_Had_PS_removed = np.take(file['amp_pileup_Had'][:], self.select)
            self.t0pil_Had_PS_removed = np.take(file['t0_pileup_Had'][:], self.select)
            self.chi2pil_Had_PS_removed = np.take(file['chi2_pileup_Had'][:], self.select)
            self.integrated_raw_PS_removed = np.take(np.array(file ['integrated_raw_amp']), self.select)    
            self.deltachi2_PS_removed = np.take(np.array(file ['Delta_chi2']), self.select)
            self.CR_PS_removed = np.take(np.array(file ['charge_ratio']), self.select)
            self.rundata_PS_removed = np.take(file['rundata'][:],self.select)
            self.waveform_number_PS_removed = np.take(file['waveform_number'][:],self.select)
            self.data_min_PS_removed = np.take(file['data_amp_min'][:],self.select)

        self.amp_EM_PS = np.delete(file ['amp_withdelay_EM'][:],self.select)
        self.t0_EM_PS = np.delete(file ['t0_withdelay_EM'][:],self.select)
        self.chi2_EM_PS = np.delete(file ['chi2_withdelay_EM'][:],self.select)
        self.amppil_EM_PS = np.delete(file['amp_pileup_EM'][:], self.select)
        self.t0pil_EM_PS = np.delete(file['t0_pileup_EM'][:], self.select)
        self.chi2pil_EM_PS = np.delete(file['chi2_pileup_EM'][:], self.select)    
        self.amp_Had_PS = np.delete(file ['amp_withdelay_Had'][:],self.select)
        self.t0_Had_PS = np.delete(file ['t0_withdelay_Had'][:],self.select)
        self.chi2_Had_PS = np.delete(file ['chi2_withdelay_Had'][:],self.select)
        self.amppil_Had_PS = np.delete(file['amp_pileup_Had'][:], self.select)
        self.t0pil_Had_PS = np.delete(file['t0_pileup_Had'][:], self.select)
        self.chi2pil_Had_PS = np.delete(file['chi2_pileup_Had'][:], self.select)
        self.integrated_raw_PS = np.delete(np.array(file ['integrated_raw_amp']), self.select)    
        self.deltachi2_PS = np.delete(np.array(file ['Delta_chi2']), self.select)
        self.CR_PS = np.delete(np.array(file ['charge_ratio']), self.select)
        self.rundata_PS = np.delete(file['rundata'][:],self.select)
        self.waveform_number_PS = np.delete(file['waveform_number'][:],self.select)
        self.data_min_PS = np.delete(file['data_amp_min'][:],self.select)

        if ctype == 'Tl' :            
            select2 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and  (abs(amppil_EM[i] )>1000 and abs(self.t0pil_EM[i]*fs )>100 ) or (abs(amppil_EM[i] )>200 and abs(self.t0pil_EM[i]*fs )>1800 )]
            select3 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and (abs(amppil_Had[i] )>700 and abs(self.t0pil_Had[i]*fs )>700 ) or (abs(amppil_Had[i] )>200 and abs(self.t0pil_Had[i]*fs )>1800 )]
            select6 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and abs(amppil_EM[i] )>20 and self.t0pil_EM[i]*fs <0]
            select7 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and abs(amppil_Had[i] )>20 and self.t0pil_Had[i]*fs <0]
            select8 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and (chi2pil_EM[i] > 3.5e7 or chi2_EM[i]/15000>3000) ]
            select9 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and (chi2pil_Had[i] > 3.5e7 or chi2_Had[i]/15000>3000) ]
            self.select_FT = list(set(list(self.select)+ select2+select3+select8+select9+ select6+select7))
            self.select_FTonly = list(set( select2+select3+ select6+select7+select8+select9))

        if ctype == 'pure':
            select2 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and  (abs(amppil_EM[i]) > 150 and  abs(self.t0pil_EM[i]*fs) > 180)]
            select3 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and (abs(amppil_Had[i] ) > 150 and abs(self.t0pil_Had[i]*fs) > 180)]
            select6 = [i for i in range (len(deltachi2)) if deltachi2[i]>0 and (abs(amppil_EM[i] )>30 and self.t0pil_EM[i]*fs <-20) or (abs(amppil_EM[i] )>50 and self.t0pil_EM[i]*fs <-10)]
            select7 = [i for i in range (len(deltachi2)) if deltachi2[i]<0 and (abs(amppil_Had[i] )>30 and self.t0pil_Had[i]*fs <-20) or (abs(amppil_Had[i] )>50 and self.t0pil_Had[i]*fs <-10)]
            self.select_FT = list(set(list(self.select)+ select3+select2+ select6+select7))
            self.select_FTonly = list(set( select2+select3+ select6+select7))

        if examine: 
            self.amp_EM_PS_FT_removed = np.take(file ['amp_withdelay_EM'][:],self.select_FT)
            self.t0_EM_PS_FT_removed = np.take(file ['t0_withdelay_EM'][:],self.select_FT)
            self.chi2_EM_PS_FT_removed = np.take(file ['chi2_withdelay_EM'][:],self.select_FT)
            self.amppil_EM_PS_FT_removed = np.take(file['amp_pileup_EM'][:], self.select_FT)
            self.t0pil_EM_PS_FT_removed = np.take(file['t0_pileup_EM'][:], self.select_FT)
            self.chi2pil_EM_PS_FT_removed = np.take(file['chi2_pileup_EM'][:], self.select_FT)    
            self.amp_Had_PS_FT_removed = np.take(file ['amp_withdelay_Had'][:],self.select_FT)
            self.t0_Had_PS_FT_removed = np.take(file ['t0_withdelay_Had'][:],self.select_FT)
            self.chi2_Had_PS_FT_removed = np.take(file ['chi2_withdelay_Had'][:],self.select_FT)
            self.amppil_Had_PS_FT_removed = np.take(file['amp_pileup_Had'][:], self.select_FT)
            self.t0pil_Had_PS_FT_removed = np.take(file['t0_pileup_Had'][:], self.select_FT)
            self.chi2pil_Had_PS_FT_removed = np.take(file['chi2_pileup_Had'][:], self.select_FT)
            self.integrated_raw_PS_FT_removed = np.take(np.array(file ['integrated_raw_amp']), self.select_FT)    
            self.deltachi2_PS_FT_removed = np.take(np.array(file ['Delta_chi2']), self.select_FT)
            self.CR_PS_FT_removed = np.take(np.array(file ['charge_ratio']), self.select_FT)
            self.rundata_PS_FT_removed = np.take(file['rundata'][:],self.select_FT)
            self.waveform_number_PS_FT_removed = np.take(file['waveform_number'][:],self.select_FT)
            self.data_min_PS_FT_removed = np.take(file['data_amp_min'][:],self.select_FT)

        self.amp_EM_PS_FT = np.delete(file ['amp_withdelay_EM'][:],self.select_FT)
        self.t0_EM_PS_FT = np.delete(file ['t0_withdelay_EM'][:],self.select_FT)
        self.chi2_EM_PS_FT = np.delete(file ['chi2_withdelay_EM'][:],self.select_FT)
        self.amppil_EM_PS_FT = np.delete(file['amp_pileup_EM'][:], self.select_FT)
        self.t0pil_EM_PS_FT = np.delete(file['t0_pileup_EM'][:], self.select_FT)
        self.chi2pil_EM_PS_FT = np.delete(file['chi2_pileup_EM'][:], self.select_FT)    
        self.amp_Had_PS_FT = np.delete(file ['amp_withdelay_Had'][:],self.select_FT)
        self.t0_Had_PS_FT = np.delete(file ['t0_withdelay_Had'][:],self.select_FT)
        self.chi2_Had_PS_FT = np.delete(file ['chi2_withdelay_Had'][:],self.select_FT)
        self.amppil_Had_PS_FT = np.delete(file['amp_pileup_Had'][:], self.select_FT)
        self.t0pil_Had_PS_FT = np.delete(file['t0_pileup_Had'][:], self.select_FT)
        self.chi2pil_Had_PS_FT = np.delete(file['chi2_pileup_Had'][:], self.select_FT)
        self.integrated_raw_PS_FT = np.delete(np.array(file ['integrated_raw_amp']), self.select_FT)    
        self.deltachi2_PS_FT = np.delete(np.array(file ['Delta_chi2']), self.select_FT)
        self.CR_PS_FT = np.delete(np.array(file ['charge_ratio']), self.select_FT)
        self.rundata_PS_FT = np.delete(file['rundata'][:],self.select_FT)
        self.waveform_number_PS_FT = np.delete(file['waveform_number'][:],self.select_FT)
        self.data_min_PS_FT = np.delete(file['data_amp_min'][:],self.select_FT)
