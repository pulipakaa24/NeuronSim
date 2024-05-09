from __future__ import division
import matplotlib
matplotlib.use('agg')  # Use a non-interactive backend

from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from scipy.integrate import odeint

app = Flask(__name__)

# Morris Lecar parameters near SNLC
phi = 0.067
g_Ca = 4
V3 = 12
V4 = 17.4
E_Ca = 120
E_K = -84
E_L = -60
g_K = 8
g_L = 2
V1 = -1.2
V2 = 18
C_M = 20

I_ext1 = 60
I_ext2 = 0

updateList = None

X0 = None

steps = 400

V0 = -10

def reset():
    global updateList
    updateList = None
    updateList = [g_Ca, E_Ca, g_K, g_L, E_K, E_L, C_M, V1, V2, I_ext1, I_ext2, V0, steps]

# ionic gates
def m_inf(V): return 0.5*(1.+np.tanh((V-updateList[7])/updateList[8]))
def tau_n(V): return 1./np.cosh((V-V3)/(2*V4))
def n_inf(V): return 0.5*(1.+np.tanh((V-V3)/V4))
    
# ionic currents
def I_leak(V): return updateList[3]*(V-updateList[5])
def I_K(V,n): return updateList[2]*n*(V-updateList[4])
def I_Ca(V): return updateList[0]*m_inf(V)*(V-updateList[1])

# neuron dynamics
def MLneuronVF(X,t,I_ext=0):
    V,n = X
    dV = (I_ext-I_leak(V)-I_K(V,n)-I_Ca(V))/updateList[6]
    dn = phi*(n_inf(V)-n)/tau_n(V)
    return dV,dn

X1 = None
X2 = None
t_span = None

def load_eqs():
    global X1, X2, X0, t_span
    # call ode solver
    X0 = [updateList[11], 0.25] # initial condition[V0,n0]
    t_span = np.arange(0.0, updateList[12], 0.1)
    X1 = odeint(lambda X,t: MLneuronVF(X,t,I_ext=updateList[9]), X0, t_span)
    X2 = odeint(lambda X,t: MLneuronVF(X,t,I_ext=updateList[10]), X0, t_span)

@app.route('/upd', methods=['POST'])
def update_plot():
    global updateList
    reset()
    data = request.form
    i = 0
    for key, value in data.items():
        updateList[i] = float(value)
        i+=1
    load_eqs()
    return generate_plot()

@app.route('/gen', methods=['GET'])
def generate_plot():
    global updateList
    # plot
    plt.figure(figsize=(10,5))
    # Voltage
    axV = plt.subplot(2,2,1)
    axV.plot(t_span,X1[:,0],'b')
    axV.plot(t_span,X2[:,0],'b--')
    axV.set_title('Membrane Potential')
    axV.set_ylabel('mV')
    plt.xlim([0,t_span[-1]])
    plt.setp(axV.get_xticklabels(), visible=False)

    # n
    axn = plt.subplot(2,2,3,sharex=axV)
    axn.plot(t_span,X1[:,1],'g')
    axn.plot(t_span,X2[:,1],'g--')
    axn.set_title('n')
    axn.set_ylabel('gate value')
    axn.set_xlabel('time (ms)')
    plt.xlim([0,t_span[-1]])

    # V,n orbit
    axP = plt.subplot(2,2,(2,4))
    axP.plot(X1[:,0],X1[:,1],'k')
    axP.plot(X2[:,0],X2[:,1],'k--')
    axP.set_xlabel('V')
    axP.set_ylabel('n')
    axP.legend(['I_ext=%.0f' %I_ext1,'I_ext=%.0f' %I_ext2])
        
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Encode plot to base64 string
    plot_url = base64.b64encode(img.getvalue()).decode()

    plot_html = f'<img src="data:image/png;base64,{plot_url}" alt="Matplotlib Plot">'
    updateList.append(plot_html)
    return updateList

@app.route('/')
def index():
    # Render template with plot
    reset()
    load_eqs()
    return render_template('index.html')

@app.route('/bg')
def bg():
    return render_template('bg.html')

if __name__ == '__main__':
    app.run(debug=True)
