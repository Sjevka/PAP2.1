# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:12:13 2018

@author: Jens Wagner
"""

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

#NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import tkinter as Tk
import datetime
from scipy.optimize import curve_fit
import peakutils
import win32api
import win32print



def set_date():
    now = datetime.datetime.now() #aktuelles Datum
    fig.text(0.76, 0.96,now.strftime("%d-%m-%Y %H:%M"))
    
root = Tk.Tk()
#root.state("zoomed") #für fullscreen
root.wm_title("coupled pendulums")
root.resizable(False,False)


fig = plt.figure(figsize=(8,10))
#Datum einfügen

ax1 = fig.add_subplot(2, 1, 1)
fig.text(0.1, 0.96,'right pendulum')
#fig.text(0.20, 0.96,entry1.get(), fontsize=18)
#ax1.set_title('right',loc='left')  
ax1.set_xlabel('time [s]')
ax1.set_ylabel('angle [a.u.]')

ax2 = fig.add_subplot(2, 1, 2)
fig.text(0.1, 0.5,'left pendulum')
#ax2.set_title('left',loc='left')  
ax2.set_xlabel('time [s]')
ax2.set_ylabel('angle [a.u.]')

set_date()

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95,
                wspace=0, hspace=0.2)

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()




def comma_to_float(valstr):
    return float(valstr.decode("utf-8").replace(',','.'))

def quit1():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
def messung_oeffnen():
    #Plot löschen    
    if plt.gcf().texts!=[]: #Title löschen (Text)
       plt.gcf().texts.clear()    
    set_date()    
    ax1.clear()
    ax2.clear() 
    #File Dialog
    root.filename = Tk.filedialog.askopenfilename(initialdir = "C:\Messungen",
    title = "Select file",filetypes = (("Pendel files","*.txt"),("all files","*.*")))
    # open file
    global zeit,p1    
    zeit,p2,p1=np.loadtxt(root.filename,skiprows=5,
                          converters={0:comma_to_float,1:comma_to_float,2:comma_to_float},
                          unpack=True)
	#plotten    
    #ax1.set_text('kkk')    
    ax1.plot(zeit,p1, antialiased=True)
    fig.text(0.1, 0.96,'right pendulum')
    #ax1.set_title('right',loc='left')  
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('angle [a.u.]')
    ax1.set_xlim([0,np.max(zeit)*1.01])
    
    ax2.plot(zeit,p2, 'r-',antialiased=True)
    fig.text(0.1, 0.5,'left pendulum')
    #ax2.set_title('left',loc='left')  
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('angle [a.u.]') 
    ax2.set_xlim([0,np.max(zeit)*1.01])
    
    fig.savefig('temp.pdf')
    canvas.draw()
    

def set_title():
    if plt.gcf().texts!=[]: #Title löschen (Text)
       plt.gcf().texts.clear()
       
    ax1.set_title(entry1.get())
    fig.text(0.1, 0.5,'right pendulum')
    fig.text(0.1, 0.96,'left pendulum')

    set_date()  
    fig.savefig('temp.pdf')
    canvas.draw()
    
def set_title2():
    ax3.set_title(entry2.get())
    #fig2.text(0.20, 0.92,entry2.get(), fontsize=18)
    fig2.savefig('fft.pdf')
    canvas2.draw()


def print_figure(filename):
    print(filename)
    win32api.ShellExecute (
    0,
    "print",
    filename,
    '/d:"%s"' % win32print.GetDefaultPrinter(),
    ".",
    0)
    
def gaussian1(t, a, mu, sig):
    return a/np.sqrt(2*np.pi)/sig*np.exp(-(t-mu)**2/(2*sig**2))

def gaussian2(t, a1, mu1, sig1, a2, mu2, sig2):
    return a1/np.sqrt(2*np.pi)/sig1*np.exp(-(t-mu1)**2/(2*sig1**2))+a2/np.sqrt(2*np.pi)/sig2*np.exp(-(t-mu2)**2/(2*sig2**2))

def fit_gaussian():
    if plt.gcf().texts!=[]: #Text löschen (Text)
       plt.gcf().texts.clear()

    fitParams, fitCovariances = curve_fit(gaussian1, freq_halb, amplitude) 
    ax3.plot(np.linspace(0.2,1.2,1000),gaussian1(np.linspace(0.2,1.2,1000),*fitParams))   
    fig2.text(.63,.58,r'$\mu = $'+str(round(fitParams[1],3))+' Hz'
    '\n'
    r'$\sigma = $'+str(round(fitParams[2],3))+' Hz'
    '\n'
    r'$FWHM = $'+str(round(fitParams[2]*2.355,3))+' Hz')
    fig2.savefig('fft.pdf')
    canvas2.draw()

    
def fit_2gaussian():    
    if plt.gcf().texts!=[]: #Text löschen (Text)
       plt.gcf().texts.clear()

    indexes = peakutils.indexes(amplitude)  # Suche Peaks für die Fitparameter
    ax3.plot(freq_halb[indexes], amplitude[indexes],marker='*', linewidth=0) 
    init_vals = [0.0002,freq_halb[indexes[0]], 0.005,0.0002, freq_halb[indexes[1]], 0.005]    
    fitParams, fitCovariances = curve_fit(gaussian2, freq_halb, amplitude,p0=init_vals) 
    ax3.plot(np.linspace(0.2,1.2,1000),gaussian2(np.linspace(0.2,1.2,1000),*fitParams))
       
    fig2.text(.63,.58,r'$\mu = $'+str(round(fitParams[1],3))+' Hz'
    '\n'
    r'$\sigma = $'+str(round(fitParams[2],3))+' Hz'
    '\n'
    r'$FWHM = $'+str(round(fitParams[2]*2.355,3))+' Hz'
    '\n'
    r'$\mu_1 = $'+str(round(fitParams[4],3))+' Hz'
    '\n'
    r'$\sigma_2 = $'+str(round(fitParams[5],3))+' Hz'
    '\n'
    r'$FWHM_2 = $'+str(round(fitParams[5]*2.355,3))+' Hz')
    canvas2.draw()
    fig2.savefig('fft.pdf')
    
    
def fft_win(): # new window definition 
#fftwin = Tk.Toplevel(root)  
    global ax3, canvas2, fig2, fftwin, entry2
    fftwin = Tk.Toplevel(root)  
    fftwin.resizable(False,False)
    button_frame3 = Tk.Frame(fftwin)
    button_frame3.pack()
    button_frame4 = Tk.Frame(fftwin)
    button_frame4.pack()
    
    button_fit_g = Tk.Button(button_frame3, text='fit gaussian', command=fit_gaussian)
    button_fit_g.pack(side=Tk.LEFT,ipadx=20)
    
    button_fit_2g = Tk.Button(button_frame3, text='fit double gaussian', command=fit_2gaussian)
    button_fit_2g.pack(side=Tk.LEFT,ipadx=20)
    
    button_print = Tk.Button(button_frame3, text='print', command= lambda: print_figure('fft.pdf'))
    button_print.pack(side=Tk.LEFT,ipadx=20)
    
    button_set_title=Tk.Button(button_frame4, text='set figure title' , command=set_title2)
    button_set_title.pack(side=Tk.LEFT)
    
    entry2=Tk.Entry(button_frame4, width=30)
    entry2.insert(0,"insert your figure title") 
    entry2.pack(side=Tk.LEFT)
    
    #FFT window:
    fig2 = plt.figure(figsize=(8,4.5))
    ax3 = fig2.add_subplot(1, 1, 1)
    canvas2 = FigureCanvasTkAgg(fig2, fftwin)
    canvas2.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    toolbar2 = NavigationToolbar2Tk(canvas2, fftwin)
    #toolbar2 = NavigationToolbar2TkAgg(canvas2, fftwin)
    toolbar2.update()  
    
    global freq_halb, amplitude 
    #Berechnung des mittleren Zeitschritts
    #Wird fuer die Berechnung der Frequenz bei FFT benoetigt 
    dt=np.array([])
    for i in range(len(zeit)-1): 
        dt=np.append( dt,zeit[i+1]-zeit[i])
    timestep=np.mean(dt)
    #Fouriertransformation mit zeropadding
    spektrum = np.fft.fft(np.concatenate((p1, np.zeros(2*len(p1)))))  
    #spektrum = np.fft.fft(p1)  #Fouriertransformation    
    freq = np.fft.fftfreq(spektrum.size, timestep)
    
   
    n=spektrum.size   #Nur positive Werte
    n_halb = np.ceil(n/2.0)
    spektrum_halb = (2.0 / int(n)) * spektrum[0:int(n_halb)]
    freq_halb = freq[0:int(n_halb)]
    amplitude=np.abs(spektrum_halb)
    
    ax3.plot(freq_halb, amplitude)
    ax3.set_xlabel("frequency /Hz")
    ax3.set_ylabel("amplitude /a.u.")
    ax3.set_xlim([0.5,0.9])
    #ax3.set_title('frequency spectrum')
    canvas2.draw()






#frame für die Buttons
button_frame = Tk.Frame(root)
button_frame.pack()

button_frame2 = Tk.Frame(root)
button_frame2.pack()


button_laden = Tk.Button(button_frame, text='open measure', command=messung_oeffnen)
button_laden.pack(side=Tk.LEFT,ipadx=20)

button_fft =Tk.Button(button_frame, text ="calculate frequency spectrum", command =fft_win) #command linked
button_fft.pack(side=Tk.LEFT)

button_print = Tk.Button(button_frame, text='print', command= lambda: print_figure('temp.pdf'))
button_print.pack(side=Tk.LEFT,ipadx=20)

button_quit = Tk.Button(button_frame, text='quit', command=quit1)
button_quit.pack(side=Tk.LEFT,ipadx=20)

canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
toolbar = NavigationToolbar2Tk(canvas, root)

#toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()

button_set_title=Tk.Button(button_frame2, text='set figure title' , command=set_title)
button_set_title.pack(side=Tk.LEFT)


entry1=Tk.Entry(button_frame2, width=30)
entry1.insert(0,"insert your figure title") 
entry1.pack(side=Tk.LEFT)

Tk.mainloop()

































