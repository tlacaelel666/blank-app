import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from qiskit import QuantumCircuit, Aer, execute
from scipy.constants import hbar, G, c

class QuantumWormholeModel:
    def __init__(self, majorana_mass=1e-22, dark_energy=-1e18, 
                 wormhole_distance=1e6, teleport_time=1e-9):
        """
        Inicializar el modelo de teletransportación a través de agujeros de gusano cuánticos
        
        Parámetros:
        - majorana_mass: Masa del fermión de Majorana (kg)
        - dark_energy: Energía negativa de la materia oscura (J)
        - wormhole_distance: Distancia del agujero de gusano (m)
        - teleport_time: Tiempo de teletransportación (s)
        """
        self.m = majorana_mass
        self.E = dark_energy
        self.d = wormhole_distance
        self.t_teleport = teleport_time
        
        # Matrices de Dirac (versión simplificada 2D)
        self.alpha_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.alpha_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.beta = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Estado inicial (simplificado)
        self.psi_initial = np.array([1, 0], dtype=complex)
        
    def dirac_equation_rhs(self, t, psi, p_x, p_y):
        """Lado derecho de la ecuación de Dirac para evolución temporal"""
        # Simplificación 2D de la ecuación de Dirac
        H_dirac = p_x * self.alpha_x + p_y * self.alpha_y + self.m * self.beta
        return (-1j / hbar) * np.dot(H_dirac, psi)
    
    def schrodinger_evolution(self, t_span, num_points=1000):
        """Evolución temporal según la ecuación de Schrödinger"""
        # Parámetros simplificados para este modelo conceptual
        p_x = hbar / self.d
        p_y = 0
        
        # Resolver la ecuación diferencial
        sol = solve_ivp(
            lambda t, y: self.dirac_equation_rhs(t, y, p_x, p_y),
            t_span,
            [self.psi_initial[0], self.psi_initial[1]],
            t_eval=np.linspace(t_span[0], t_span[1], num_points),
            method='RK45'
        )
        
        return sol.t, np.array([sol.y[0], sol.y[1]])
    
    def einstein_field_equation(self, r):
        """
        Implementación simplificada de la ecuación de campo de Einstein
        para modelar la métrica del agujero de gusano
        """
        # Constantes relevantes para la ecuación de Einstein
        k = 8 * np.pi * G / c**4
        
        # Densidad de energía del agujero de gusano (simplificación)
        rho = self.E / (4/3 * np.pi * r**3)
        
        # Componente temporal del tensor de Ricci (simplificación)
        R_00 = -k * (rho + 3 * self.p(r))
        
        return R_00
    
    def p(self, r):
        """Presión en función de la distancia (simplificación)"""
        return -self.E / (4 * np.pi * r**3 * c**2)
    
    def wormhole_metric(self, r_values):
        """Cálculo simplificado de la métrica del agujero de gusano"""
        # Modelo de métrica de Morris-Thorne simplificado
        b = 2 * G * self.m / c**2  # Radio de garganta simplificado
        
        # Factor de forma del agujero de gusano
        phi = np.zeros_like(r_values)
        g_tt = -np.exp(2 * phi)
        g_rr = 1 / (1 - b / r_values)
        
        return g_tt, g_rr
    
    def quantum_teleportation_circuit(self):
        """
        Crea un circuito cuántico que representa conceptualmente
        la teletransportación a través del agujero de gusano
        """
        # Crear un circuito de 3 qubits
        qc = QuantumCircuit(3, 2)
        
        # Preparar el estado a teleportar (qubit 0)
        qc.h(0)  # Estado superposición
        
        # Crear entrelazamiento entre los extremos del agujero de gusano (qubits 1 y 2)
        qc.h(1)
        qc.cx(1, 2)
        
        # Interacción del estado con la entrada del agujero de gusano
        qc.cx(0, 1)
        qc.h(0)
        
        # Medición (colapso del espacio-tiempo en el modelo conceptual)
        qc.measure([0, 1], [0, 1])
        
        # Correcciones basadas en las mediciones (salida del agujero de gusano)
        qc.cx(1, 2)
        qc.cz(0, 2)
        
        return qc
    
    def run_simulation(self):
        """Ejecuta la simulación completa del modelo"""
        results = {}
        
        # 1. Evolución de la función de onda del fermión de Majorana
        t_span = (0, self.t_teleport)
        t_values, psi_values = self.schrodinger_evolution(t_span)
        results['wave_function'] = {
            't': t_values,
            'psi': psi_values,
            'probability': np.abs(psi_values)**2
        }
        
        # 2. Cálculo de la métrica del agujero de gusano
        r_values = np.linspace(self.d/100, self.d, 1000)
        g_tt, g_rr = self.wormhole_metric(r_values)
        results['metric'] = {
            'r': r_values,
            'g_tt': g_tt,
            'g_rr': g_rr
        }
        
        # 3. Simular el circuito cuántico de teletransportación
        qc = self.quantum_teleportation_circuit()
        simulator = Aer.get_backend('statevector_simulator')
        job = execute(qc, simulator)
        state_vector = job.result().get_statevector()
        results['quantum_circuit'] = {
            'circuit': qc,
            'state_vector': state_vector
        }
        
        return results
    
    def visualize_results(self, results):
        """Visualizar los resultados de la simulación"""
        # Configurar el estilo de las gráficas
        plt.style.use('seaborn-darkgrid')
        
        # Crear una figura con tres subfiguras
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # 1. Evolución de la función de onda
        axs[0].plot(results['wave_function']['t'], 
                   np.abs(results['wave_function']['psi'][0])**2, 
                   label='|ψ₁|²')
        axs[0].plot(results['wave_function']['t'], 
                   np.abs(results['wave_function']['psi'][1])**2, 
                   label='|ψ₂|²')
        axs[0].set_xlabel('Tiempo (s)')
        axs[0].set_ylabel('Probabilidad')
        axs[0].set_title('Evolución de la función de onda del fermión de Majorana')
        axs[0].legend()
        
        # 2. Métrica del agujero de gusano
        axs[1].plot(results['metric']['r'], results['metric']['g_tt'], 
                   label='g₍ₜₜ₎')
        axs[1].plot(results['metric']['r'], results['metric']['g_rr'], 
                   label='g₍ᵣᵣ₎')
        axs[1].set_xlabel('Distancia radial (m)')
        axs[1].set_ylabel('Componentes de la métrica')
        axs[1].set_title('Métrica del agujero de gusano')
        axs[1].legend()
        
        # 3. Representación de las probabilidades finales
        probs = np.abs(results['quantum_circuit']['state_vector'])**2
        states = [f"|{i:03b}⟩" for i in range(8)]
        axs[2].bar(states, probs)
        axs[2].set_xlabel('Estados')
        axs[2].set_ylabel('Probabilidad')
        axs[2].set_title('Distribución de probabilidad después de la teletransportación')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig

# Ejemplo de uso del modelo
def run_wormhole_teleportation_simulation():
    # Crear instancia del modelo con los parámetros especificados
    model = QuantumWormholeModel(
        majorana_mass=1e-22,  # kg
        dark_energy=-1e18,    # J
        wormhole_distance=1e6, # m
        teleport_time=1e-9     # s
    )
    
    # Ejecutar la simulación
    print("Iniciando simulación del modelo de teletransportación cuántica...")
    results = model.run_simulation()
    
    # Visualizar resultados
    fig = model.visualize_results(results)
    
    print("Simulación completada.")
    return model, results, fig

if __name__ == "__main__":
    model, results, fig = run_wormhole_teleportation_simulation()
    plt.show()
    
    # Mostrar detalles del circuito cuántico
    print("\nCircuito de teletransportación cuántica:")
    print(results['quantum_circuit']['circuit'])