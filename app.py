import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import math

def format_brl(value):
    """Format currency in Brazilian Real standard (R$ X.XXX,XX)"""
    # Format with 2 decimal places
    formatted = f"{value:.2f}"
    
    # Split into integer and decimal parts
    parts = formatted.split(".")
    integer_part = parts[0]
    decimal_part = parts[1]
    
    # Add thousands separators (periods) to integer part
    if len(integer_part) > 3:
        # Reverse, add dots every 3 digits, then reverse back
        reversed_int = integer_part[::-1]
        chunks = [reversed_int[i:i+3] for i in range(0, len(reversed_int), 3)]
        integer_part = ".".join(chunks)[::-1]
    
    # Return in Brazilian format: R$ X.XXX,XX
    return f"R$ {integer_part},{decimal_part}"

# Configuração da página
st.set_page_config(
    page_title="Simulador Microeconômico - Teoria da Firma",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Simulador Microeconômico - Teoria da Firma")
st.markdown("---")

# Sidebar para controles de parâmetros
st.sidebar.header("⚙️ Parâmetros da Firma")
st.sidebar.markdown("Ajuste os parâmetros abaixo para simular diferentes cenários econômicos:")

# Parâmetros de custo
st.sidebar.subheader("💰 Função de Custo")
fixed_cost = st.sidebar.slider("Custo Fixo (CF)", min_value=0, max_value=1000, value=100, step=10,
                               help="Custos que não variam com a produção (ex: aluguel, equipamentos)")

variable_cost_linear = st.sidebar.slider("Custo Variável Linear", min_value=0.0, max_value=50.0, value=10.0, step=0.5,
                                        help="Componente linear do custo variável por unidade")

variable_cost_quadratic = st.sidebar.slider("Custo Variável Quadrático", min_value=0.0, max_value=5.0, value=0.5, step=0.1,
                                           help="Componente quadrático do custo variável (retornos decrescentes)")

# Parâmetros de receita
st.sidebar.subheader("💵 Função de Receita")
market_structure = st.sidebar.selectbox("Estrutura de Mercado", 
                                       ["Concorrência Perfeita", "Monopólio"],
                                       help="Tipo de mercado que determina a função de receita")

if market_structure == "Concorrência Perfeita":
    price = st.sidebar.slider("Preço de Mercado (P)", min_value=1.0, max_value=100.0, value=25.0, step=0.5,
                             help="Preço fixo determinado pelo mercado")
    demand_intercept = price
    demand_slope = 0
else:  # Monopólio
    demand_intercept = st.sidebar.slider("Intercepto da Demanda", min_value=10.0, max_value=200.0, value=50.0, step=1.0,
                                        help="Preço máximo que os consumidores estão dispostos a pagar")
    demand_slope = st.sidebar.slider("Inclinação da Demanda", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                    help="Sensibilidade do preço à quantidade (elasticidade)")

# Parâmetros de produção
st.sidebar.subheader("🏭 Parâmetros de Produção")
max_quantity = st.sidebar.slider("Quantidade Máxima", min_value=10, max_value=200, value=100, step=5,
                                help="Limite máximo de produção para análise")

# Simulador de cenários
st.sidebar.markdown("---")
st.sidebar.subheader("🎭 Simulador de Cenários")
st.sidebar.markdown("Explore cenários econômicos pré-definidos:")

scenario_options = [
    "Configuração Manual",
    "Startup Tecnológica",
    "Fábrica Tradicional", 
    "Empresa de Serviços",
    "Indústria Farmacêutica",
    "Restaurante Local"
]

selected_scenario = st.sidebar.selectbox(
    "Escolha um Cenário:",
    scenario_options,
    help="Selecione um cenário pré-configurado ou mantenha 'Configuração Manual'"
)

# Configurações de cenários predefinidos
if selected_scenario != "Configuração Manual":
    if selected_scenario == "Startup Tecnológica":
        # Alto custo fixo (investimento em tecnologia), baixo custo variável
        fixed_cost = 800
        variable_cost_linear = 5.0
        variable_cost_quadratic = 0.2
        if market_structure == "Monopólio":
            demand_intercept = 80.0
            demand_slope = 0.8
        else:
            price = 35.0
        max_quantity = 150
        
    elif selected_scenario == "Fábrica Tradicional":
        # Custo fixo moderado, custo variável crescente (mão de obra intensiva)
        fixed_cost = 400
        variable_cost_linear = 15.0
        variable_cost_quadratic = 0.8
        if market_structure == "Monopólio":
            demand_intercept = 60.0
            demand_slope = 1.2
        else:
            price = 28.0
        max_quantity = 120
        
    elif selected_scenario == "Empresa de Serviços":
        # Baixo custo fixo, custo variável linear alto (baseado em tempo)
        fixed_cost = 150
        variable_cost_linear = 20.0
        variable_cost_quadratic = 0.3
        if market_structure == "Monopólio":
            demand_intercept = 70.0
            demand_slope = 1.5
        else:
            price = 32.0
        max_quantity = 100
        
    elif selected_scenario == "Indústria Farmacêutica":
        # Muito alto custo fixo (P&D), baixo custo variável marginal
        fixed_cost = 1200
        variable_cost_linear = 3.0
        variable_cost_quadratic = 0.1
        if market_structure == "Monopólio":
            demand_intercept = 120.0
            demand_slope = 0.6
        else:
            price = 45.0
        max_quantity = 180
        
    elif selected_scenario == "Restaurante Local":
        # Custo fixo médio, custo variável moderado com crescimento rápido
        fixed_cost = 300
        variable_cost_linear = 12.0
        variable_cost_quadratic = 1.2
        if market_structure == "Monopólio":
            demand_intercept = 45.0
            demand_slope = 2.0
        else:
            price = 22.0
        max_quantity = 80
    
    # Atualizar valores no sidebar para mostrar as configurações aplicadas
    st.sidebar.success(f"✅ Cenário '{selected_scenario}' aplicado!")
    st.sidebar.info(f"""
    **Configurações aplicadas:**
    • Custo Fixo: R$ {fixed_cost}
    • Custo Var. Linear: R$ {variable_cost_linear}
    • Custo Var. Quadrático: {variable_cost_quadratic}
    • Qtd. Máxima: {max_quantity}
    """)
    
    if market_structure == "Monopólio":
        st.sidebar.info(f"""
        **Parâmetros de Demanda:**
        • Intercepto: R$ {demand_intercept}
        • Inclinação: {demand_slope}
        """)
    else:
        st.sidebar.info(f"""
        **Preço de Mercado:**
        • Preço: R$ {price}
        """)
else:
    # Se configuração manual, usar os valores dos sliders originais
    pass

# Funções econômicas
def calculate_costs(q, fc, vc_linear, vc_quad):
    """Calcula funções de custo"""
    # Custo Total: CT = CF + CVL*Q + CVQ*Q²
    total_cost = fc + vc_linear * q + vc_quad * (q ** 2)
    
    # Custo Marginal: CMg = dCT/dQ = CVL + 2*CVQ*Q
    marginal_cost = vc_linear + 2 * vc_quad * q
    
    # Custo Médio: CMe = CT/Q
    average_cost = np.where(q > 0, total_cost / q, np.inf)
    
    # Custo Variável Médio: CVMe = CV/Q
    variable_cost = vc_linear * q + vc_quad * (q ** 2)
    avg_variable_cost = np.where(q > 0, variable_cost / q, np.inf)
    
    return total_cost, marginal_cost, average_cost, avg_variable_cost, variable_cost

def calculate_revenue(q, market_type, p=None, intercept=None, slope=None):
    """Calcula funções de receita"""
    if market_type == "Concorrência Perfeita":
        # Receita Total: RT = P * Q
        total_revenue = p * q
        # Receita Marginal: RMg = P (constante)
        marginal_revenue = np.full_like(q, p)
        # Preço
        price_curve = np.full_like(q, p)
    else:  # Monopólio
        # Função de demanda: P = a - b*Q
        price_curve = intercept - slope * q
        # Receita Total: RT = P * Q = (a - b*Q) * Q = a*Q - b*Q²
        total_revenue = price_curve * q
        # Receita Marginal: RMg = dRT/dQ = a - 2*b*Q
        marginal_revenue = intercept - 2 * slope * q
    
    return total_revenue, marginal_revenue, price_curve

# Gerar dados
quantities = np.linspace(0, max_quantity, 1000)
quantities_nonzero = np.linspace(0.1, max_quantity, 1000)  # Para evitar divisão por zero

# Check if we need to recalculate with employee costs
if 'num_employees' in st.session_state:
    employee_fixed_cost_main = (st.session_state.num_employees - 1) * 4000
    adjusted_fixed_cost_main = fixed_cost + employee_fixed_cost_main
else:
    adjusted_fixed_cost_main = fixed_cost

# Calcular custos
tc, mc, ac, avc, vc = calculate_costs(quantities, adjusted_fixed_cost_main, variable_cost_linear, variable_cost_quadratic)
_, mc_nz, ac_nz, avc_nz, _ = calculate_costs(quantities_nonzero, adjusted_fixed_cost_main, variable_cost_linear, variable_cost_quadratic)

# Calcular receitas
if market_structure == "Concorrência Perfeita":
    tr, mr, price_curve = calculate_revenue(quantities, market_structure, p=price)
else:
    tr, mr, price_curve = calculate_revenue(quantities, market_structure, intercept=demand_intercept, slope=demand_slope)

# Calcular lucro
profit = tr - tc

# Encontrar ponto de maximização do lucro
profit_valid = profit[quantities > 0]
quantities_valid = quantities[quantities > 0]

if len(profit_valid) > 0:
    max_profit_idx = np.argmax(profit_valid)
    optimal_quantity = quantities_valid[max_profit_idx]
    max_profit = profit_valid[max_profit_idx]
    
    # Calcular valores no ponto ótimo
    optimal_tc, optimal_mc, optimal_ac, optimal_avc, _ = calculate_costs(optimal_quantity, adjusted_fixed_cost_main, variable_cost_linear, variable_cost_quadratic)
    if market_structure == "Concorrência Perfeita":
        optimal_tr, optimal_mr, optimal_price = calculate_revenue(optimal_quantity, market_structure, p=price)
    else:
        optimal_tr, optimal_mr, optimal_price = calculate_revenue(optimal_quantity, market_structure, intercept=demand_intercept, slope=demand_slope)
    
    # Verificar shutdown point - se preço < CVMe, a firma deve parar a produção
    shutdown_condition = False
    if market_structure == "Concorrência Perfeita":
        # Em concorrência perfeita, comparar preço de mercado com CVMe
        if price < optimal_avc and optimal_quantity > 0:
            shutdown_condition = True
    else:
        # Em monopólio, comparar preço ótimo com CVMe
        if optimal_price < optimal_avc and optimal_quantity > 0:
            shutdown_condition = True
    
    # Se deve parar a produção, ajustar valores
    if shutdown_condition:
        shutdown_quantity = optimal_quantity
        shutdown_avc = optimal_avc
        shutdown_price = optimal_price
        # No shutdown, produção = 0, mas custos fixos permanecem
        optimal_quantity = 0
        max_profit = -adjusted_fixed_cost_main  # Só os custos fixos
        optimal_tc = adjusted_fixed_cost_main
        optimal_mc = optimal_ac = optimal_avc = 0
        optimal_tr = optimal_mr = optimal_price = 0
else:
    optimal_quantity = 0
    max_profit = 0
    optimal_tc = optimal_mc = optimal_ac = optimal_avc = 0
    optimal_tr = optimal_mr = optimal_price = 0
    shutdown_condition = False

# Layout principal com colunas
col1, col2 = st.columns([2, 1])

with col1:
    # Gráficos principais
    
    # Gráfico 1: Funções de Custo
    st.subheader("📈 Funções de Custo")
    
    fig_cost = go.Figure()
    
    # Custo Total
    fig_cost.add_trace(go.Scatter(
        x=quantities, y=tc,
        mode='lines',
        name='Custo Total (CT)',
        line=dict(color='red', width=3),
        hovertemplate='Quantidade: %{x:.1f}<br>Custo Total: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Custo Marginal
    fig_cost.add_trace(go.Scatter(
        x=quantities_nonzero, y=mc_nz,
        mode='lines',
        name='Custo Marginal (CMg)',
        line=dict(color='orange', width=2),
        hovertemplate='Quantidade: %{x:.1f}<br>Custo Marginal: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Custo Médio
    fig_cost.add_trace(go.Scatter(
        x=quantities_nonzero, y=ac_nz,
        mode='lines',
        name='Custo Médio (CMe)',
        line=dict(color='blue', width=2),
        hovertemplate='Quantidade: %{x:.1f}<br>Custo Médio: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Custo Variável Médio
    fig_cost.add_trace(go.Scatter(
        x=quantities_nonzero, y=avc_nz,
        mode='lines',
        name='Custo Variável Médio (CVMe)',
        line=dict(color='purple', width=2),
        hovertemplate='Quantidade: %{x:.1f}<br>CVMe: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Ponto ótimo
    if optimal_quantity > 0:
        fig_cost.add_trace(go.Scatter(
            x=[optimal_quantity], y=[optimal_mc],
            mode='markers',
            name='Ponto Ótimo',
            marker=dict(color='green', size=12, symbol='diamond'),
            hovertemplate='Quantidade Ótima: %{x:.1f}<br>CMg = RMg: R$ %{y:.2f}<extra></extra>'
        ))
    
    fig_cost.update_layout(
        title="Funções de Custo da Firma",
        xaxis_title="Quantidade (Q)",
        yaxis_title="Valor (R$)",
        hovermode='closest',
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # Gráfico 2: Funções de Receita e Lucro
    st.subheader("💰 Receita e Maximização do Lucro")
    
    fig_revenue = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Receita Total vs Custo Total', 'Lucro Total'),
        vertical_spacing=0.15
    )
    
    # Subplot 1: Receita e Custo Total
    fig_revenue.add_trace(go.Scatter(
        x=quantities, y=tr,
        mode='lines',
        name='Receita Total (RT)',
        line=dict(color='green', width=3),
        hovertemplate='Quantidade: %{x:.1f}<br>Receita Total: R$ %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig_revenue.add_trace(go.Scatter(
        x=quantities, y=tc,
        mode='lines',
        name='Custo Total (CT)',
        line=dict(color='red', width=3),
        hovertemplate='Quantidade: %{x:.1f}<br>Custo Total: R$ %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Subplot 2: Lucro
    fig_revenue.add_trace(go.Scatter(
        x=quantities, y=profit,
        mode='lines',
        name='Lucro (π)',
        line=dict(color='blue', width=3),
        fill='tonexty',
        hovertemplate='Quantidade: %{x:.1f}<br>Lucro: R$ %{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Linha de lucro zero
    fig_revenue.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Ponto de maximização do lucro
    if optimal_quantity > 0:
        fig_revenue.add_trace(go.Scatter(
            x=[optimal_quantity], y=[max_profit],
            mode='markers',
            name='Lucro Máximo',
            marker=dict(color='gold', size=15, symbol='star'),
            hovertemplate='Quantidade Ótima: %{x:.1f}<br>Lucro Máximo: R$ %{y:.2f}<extra></extra>'
        ), row=2, col=1)
    
    fig_revenue.update_layout(
        title="Análise de Receita e Lucro",
        height=600,
        showlegend=True
    )
    
    fig_revenue.update_xaxes(title_text="Quantidade (Q)", row=2, col=1)
    fig_revenue.update_yaxes(title_text="Valor (R$)", row=1, col=1)
    fig_revenue.update_yaxes(title_text="Lucro (R$)", row=2, col=1)
    
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Gráfico 3: Análise Marginal
    st.subheader("🎯 Análise Marginal - Condição de Primeira Ordem")
    
    fig_marginal = go.Figure()
    
    # Receita Marginal
    fig_marginal.add_trace(go.Scatter(
        x=quantities, y=mr,
        mode='lines',
        name='Receita Marginal (RMg)',
        line=dict(color='green', width=3),
        hovertemplate='Quantidade: %{x:.1f}<br>RMg: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Custo Marginal
    fig_marginal.add_trace(go.Scatter(
        x=quantities_nonzero, y=mc_nz,
        mode='lines',
        name='Custo Marginal (CMg)',
        line=dict(color='red', width=3),
        hovertemplate='Quantidade: %{x:.1f}<br>CMg: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Preço (se aplicável)
    if market_structure == "Monopólio":
        fig_marginal.add_trace(go.Scatter(
            x=quantities, y=price_curve,
            mode='lines',
            name='Preço (Demanda)',
            line=dict(color='blue', width=2, dash='dash'),
            hovertemplate='Quantidade: %{x:.1f}<br>Preço: R$ %{y:.2f}<extra></extra>'
        ))
    
    # Ponto de interseção RMg = CMg
    if optimal_quantity > 0:
        fig_marginal.add_trace(go.Scatter(
            x=[optimal_quantity], y=[optimal_mr],
            mode='markers',
            name='RMg = CMg',
            marker=dict(color='purple', size=15, symbol='cross'),
            hovertemplate='Quantidade Ótima: %{x:.1f}<br>RMg = CMg: R$ %{y:.2f}<extra></extra>'
        ))
        
        # Linha vertical no ponto ótimo
        fig_marginal.add_vline(
            x=optimal_quantity, 
            line_dash="dot", 
            line_color="purple",
            annotation_text=f"Q* = {optimal_quantity:.1f}"
        )
    
    fig_marginal.update_layout(
        title="Condição de Maximização: Receita Marginal = Custo Marginal",
        xaxis_title="Quantidade (Q)",
        yaxis_title="Valor (R$)",
        hovermode='closest',
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig_marginal, use_container_width=True)

with col2:
    # Painel de resultados
    st.subheader("📊 Resultados da Análise")
    
    # Métricas principais
    st.metric("Quantidade Ótima", f"{optimal_quantity:.2f} unidades")
    st.metric("Lucro Máximo", f"R$ {max_profit:.2f}", delta=f"{max_profit:.2f}")
    if market_structure == "Concorrência Perfeita":
        st.metric("Preço de Venda (Mercado)", f"R$ {optimal_price:.2f}")
    else:
        st.metric("Preço Ótimo", f"R$ {optimal_price:.2f}")
    
    st.markdown("---")
    
    # Detalhes no ponto ótimo
    st.subheader("🎯 No Ponto Ótimo:")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Custos:**")
        st.write(f"• CT: R$ {optimal_tc:.2f}")
        st.write(f"• CMg: R$ {optimal_mc:.2f}")
        st.write(f"• CMe: R$ {optimal_ac:.2f}")
    
    with col_b:
        st.write("**Receitas:**")
        st.write(f"• RT: R$ {optimal_tr:.2f}")
        st.write(f"• RMg: R$ {optimal_mr:.2f}")
        if market_structure == "Concorrência Perfeita":
            st.write(f"• Preço de Venda (Mercado): R$ {optimal_price:.2f}")
        else:
            st.write(f"• Preço: R$ {optimal_price:.2f}")
    
    st.markdown("---")
    
    # Crisis and Success Simulator
    st.subheader("⚡ Simulador de Crise e Sucesso")
    st.markdown("Demonstre os efeitos da alavancagem operacional:")
    
    col_crisis, col_success = st.columns(2)
    
    with col_crisis:
        if st.button("🔻 Simular Crise 25%", use_container_width=True, type="secondary"):
            if optimal_quantity > 0:
                # Reduzir quantidade ótima em 25%
                crisis_quantity = optimal_quantity * 0.75
                
                # Calcular novos valores
                crisis_tc, _, _, _, _ = calculate_costs(crisis_quantity, adjusted_fixed_cost_main, variable_cost_linear, variable_cost_quadratic)
                if market_structure == "Concorrência Perfeita":
                    crisis_tr, _, crisis_price = calculate_revenue(crisis_quantity, market_structure, p=price)
                else:
                    crisis_tr, _, crisis_price = calculate_revenue(crisis_quantity, market_structure, intercept=demand_intercept, slope=demand_slope)
                
                crisis_profit = crisis_tr - crisis_tc
                profit_change = crisis_profit - max_profit
                profit_change_pct = (profit_change / max_profit * 100) if max_profit != 0 else 0
                
                st.error("📉 **Cenário de Crise (-25%)**")
                st.write(f"• Nova Qtd: {crisis_quantity:.2f} unidades")
                st.write(f"• Novo Lucro: R$ {crisis_profit:.2f}")
                st.write(f"• Variação: R$ {profit_change:.2f}")
                st.write(f"• Variação %: {profit_change_pct:.1f}%")
                
                # Show operating leverage effect
                operating_leverage = abs(profit_change_pct / 25.0)
                st.info(f"🔍 **Alavancagem Operacional: {operating_leverage:.2f}x**")
                st.caption("Para cada 1% de redução nas vendas, o lucro diminui em {:.1f}%".format(operating_leverage))
    
    with col_success:
        if st.button("🔺 Simular Sucesso 25%", use_container_width=True, type="primary"):
            if optimal_quantity > 0:
                # Aumentar quantidade ótima em 25%
                success_quantity = optimal_quantity * 1.25
                
                # Calcular novos valores
                success_tc, _, _, _, _ = calculate_costs(success_quantity, adjusted_fixed_cost_main, variable_cost_linear, variable_cost_quadratic)
                if market_structure == "Concorrência Perfeita":
                    success_tr, _, success_price = calculate_revenue(success_quantity, market_structure, p=price)
                else:
                    success_tr, _, success_price = calculate_revenue(success_quantity, market_structure, intercept=demand_intercept, slope=demand_slope)
                
                success_profit = success_tr - success_tc
                profit_change = success_profit - max_profit
                profit_change_pct = (profit_change / max_profit * 100) if max_profit != 0 else 0
                
                st.success("📈 **Cenário de Sucesso (+25%)**")
                st.write(f"• Nova Qtd: {success_quantity:.2f} unidades")
                st.write(f"• Novo Lucro: R$ {success_profit:.2f}")
                st.write(f"• Variação: R$ {profit_change:.2f}")
                st.write(f"• Variação %: {profit_change_pct:.1f}%")
                
                # Show operating leverage effect
                operating_leverage = abs(profit_change_pct / 25.0)
                st.info(f"🔍 **Alavancagem Operacional: {operating_leverage:.2f}x**")
                st.caption("Para cada 1% de aumento nas vendas, o lucro aumenta em {:.1f}%".format(operating_leverage))
    
    st.markdown("---")
    
    # Growth and Diminishing Returns Simulator
    st.subheader("📈 Simulador de Crescimento e Rendimentos Decrescentes")
    st.markdown("Simule a contratação de funcionários e observe os efeitos na capacidade produtiva:")
    
    # Initialize session state for employees if not exists
    if 'num_employees' not in st.session_state:
        st.session_state.num_employees = 1
    
    # Calculate employee impact on fixed costs and production capacity
    employee_fixed_cost = (st.session_state.num_employees - 1) * 4000  # R$ 4,000 per additional employee
    adjusted_fixed_cost = fixed_cost + employee_fixed_cost
    
    # Calculate production capacity with specific diminishing returns logic
    def calculate_total_capacity(num_employees):
        total_cap = 0
        for i in range(1, num_employees + 1):
            if i == 1:
                total_cap += 100  # 1st employee adds 100 units
            elif i == 2:
                total_cap += 80   # 2nd employee adds 80 units
            elif i == 3:
                total_cap += 60   # 3rd employee adds 60 units
            elif i == 4:
                total_cap += 40   # 4th employee adds 40 units
            else:
                total_cap += 20   # 5th+ employees add 20 units each
        return total_cap
    
    total_capacity = calculate_total_capacity(st.session_state.num_employees)
    
    col_hire, col_fire = st.columns(2)
    
    with col_hire:
        if st.button("👨‍💼 Contratar Funcionário", use_container_width=True, type="primary"):
            if st.session_state.num_employees < 20:  # Limit to prevent extreme values
                st.session_state.num_employees += 1
                # Calculate marginal contribution of new employee
                if st.session_state.num_employees == 2:
                    marginal_add = 80
                elif st.session_state.num_employees == 3:
                    marginal_add = 60
                elif st.session_state.num_employees == 4:
                    marginal_add = 40
                else:
                    marginal_add = 20
                st.success(f"Funcionário contratado! (+{marginal_add} unidades, +R$ 4.000 custos fixos)")
                st.rerun()  # Force immediate update
    
    with col_fire:
        if st.button("❌ Demitir Funcionário", use_container_width=True, type="secondary"):
            if st.session_state.num_employees > 1:  # Always keep at least 1 employee
                # Calculate what we're losing
                if st.session_state.num_employees == 2:
                    marginal_loss = 80
                elif st.session_state.num_employees == 3:
                    marginal_loss = 60
                elif st.session_state.num_employees == 4:
                    marginal_loss = 40
                else:
                    marginal_loss = 20
                st.session_state.num_employees -= 1
                st.error(f"Funcionário demitido. (-{marginal_loss} unidades, -R$ 4.000 custos fixos)")
                st.rerun()  # Force immediate update
    
    # Calculate marginal productivity (additional output from last employee)
    if st.session_state.num_employees > 1:
        prev_capacity = calculate_total_capacity(st.session_state.num_employees - 1)
        marginal_productivity = total_capacity - prev_capacity
    else:
        prev_capacity = 0
        marginal_productivity = total_capacity
    
    # Display current status
    col_emp, col_cap = st.columns(2)
    
    with col_emp:
        st.metric("Funcionários Atuais", st.session_state.num_employees)
    
    with col_cap:
        st.metric("Capacidade Total de Produção (unidades)", f"{total_capacity}")
    
    # Show cost impact
    if st.session_state.num_employees > 1:
        num_new_employees = st.session_state.num_employees - 1
        st.info(f"💰 **Impacto:** +{format_brl(employee_fixed_cost)} nos custos fixos  \n**(Contratação de {num_new_employees} {'novo funcionário' if num_new_employees == 1 else 'novos funcionários'})**")
        st.write(f"**Custo Fixo Total:** {format_brl(adjusted_fixed_cost)} (original: {format_brl(fixed_cost)})")
    else:
        st.info("💰 **Custo Fixo:** Apenas o custo fixo base (sem funcionários extras)")
    
    # Show diminishing returns explanation
    if st.session_state.num_employees > 1:
        # Calculate previous employee's marginal productivity
        if st.session_state.num_employees > 2:
            prev_prev_capacity = calculate_total_capacity(st.session_state.num_employees - 2)
            prev_marginal = prev_capacity - prev_prev_capacity
        else:
            prev_marginal = 100  # First employee adds 100
        
        if marginal_productivity < prev_marginal:
            st.info("📉 **Rendimentos Decrescentes:** O último funcionário contratado produziu menos que o anterior, demonstrando a lei dos rendimentos marginais decrescentes.")
        else:
            st.info("📊 **Análise:** Observe como a produtividade marginal muda conforme você contrata mais funcionários.")
    
    # Visual representation of diminishing returns
    if st.session_state.num_employees >= 2:
        employees_range = list(range(1, st.session_state.num_employees + 1))
        capacities = [calculate_total_capacity(emp) for emp in employees_range]
        marginal_prods = []
        for i, emp in enumerate(employees_range):
            if emp == 1:
                marginal_prods.append(100)  # 1st employee adds 100
            elif emp == 2:
                marginal_prods.append(80)   # 2nd employee adds 80
            elif emp == 3:
                marginal_prods.append(60)   # 3rd employee adds 60
            elif emp == 4:
                marginal_prods.append(40)   # 4th employee adds 40
            else:
                marginal_prods.append(20)   # 5th+ employees add 20
        
        fig_diminishing = go.Figure()
        
        # Total production capacity
        fig_diminishing.add_trace(go.Scatter(
            x=employees_range, y=capacities,
            mode='lines+markers',
            name='Capacidade Total',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Marginal productivity
        fig_diminishing.add_trace(go.Scatter(
            x=employees_range, y=marginal_prods,
            mode='lines+markers',
            name='Produtividade Marginal',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig_diminishing.update_layout(
            title="Lei dos Rendimentos Decrescentes",
            xaxis=dict(
                title="Número de Funcionários",
                tickmode='linear',
                tick0=1,
                dtick=1,  # Show only integer values
                showgrid=True
            ),
            yaxis=dict(
                title="Capacidade Total (unidades)",
                showgrid=True,
                side='left'
            ),
            yaxis2=dict(
                title="Produtividade Marginal (unidades)",
                overlaying='y',
                side='right',
                color='red',
                showgrid=False
            ),
            height=400,
            showlegend=True,
            legend=dict(
                x=1.02,  # Position outside the plot area to the right
                y=1.0,   # At the top
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.3)',
                borderwidth=1,
                xanchor='left',  # Anchor the legend box to its left edge
                yanchor='top',   # Anchor the legend box to its top edge
                itemsizing='constant',  # Keep consistent spacing
                itemwidth=30,    # Add space between marker and text
                tracegroupgap=8  # Add vertical gap between legend items
            ),
            margin=dict(l=80, r=150, t=50, b=50)  # Increase right margin for legend space
        )
        
        st.plotly_chart(fig_diminishing, use_container_width=True)
    
    st.markdown("---")
    
    # Interpretação econômica
    st.subheader("💡 Interpretação Econômica")
    
    # Verificar shutdown condition primeiro
    if shutdown_condition:
        st.error("🚫 **SHUTDOWN POINT - PARAR PRODUÇÃO**")
        st.write(f"O preço (R$ {shutdown_price:.2f}) é menor que o Custo Variável Médio (R$ {shutdown_avc:.2f}).")
        st.write("**Decisão:** A firma deve parar a produção no curto prazo.")
        st.write("**Motivo:** Não consegue cobrir nem os custos variáveis.")
        st.write(f"**Prejuízo com shutdown:** R$ {max_profit:.2f} (apenas custos fixos)")
        st.write(f"**Prejuízo se continuasse produzindo:** R$ {shutdown_price * shutdown_quantity - calculate_costs(shutdown_quantity, adjusted_fixed_cost_main, variable_cost_linear, variable_cost_quadratic)[0]:.2f}")
    elif max_profit > 0:
        st.success("✅ **Lucro Econômico Positivo**")
        st.write("A firma está obtendo lucros superiores ao custo de oportunidade.")
    elif max_profit == 0:
        st.info("⚖️ **Lucro Econômico Zero**")
        st.write("A firma está no ponto de equilíbrio, cobrindo todos os custos.")
    else:
        st.error("❌ **Prejuízo Econômico**")
        st.write("A firma está operando com prejuízo, mas ainda cobre os custos variáveis.")
        st.write("**Decisão:** Continuar produzindo no curto prazo, considerar saída no longo prazo.")
    
    # Warning adicional para shutdown point (conforme solicitado)
    if shutdown_condition:
        st.warning("ALERTA: PONTO DE FECHAMENTO. O preço de mercado não cobre os custos variáveis. A firma minimizaria suas perdas paralisando a produção no curto prazo.")
    
    # Análise de mercado
    if market_structure == "Concorrência Perfeita":
        st.write("**Concorrência Perfeita:**")
        st.write("• Firma é tomadora de preços")
        st.write("• P = RMg = CMg no ótimo")
        st.write("• Eficiência alocativa")
    else:
        st.write("**Monopólio:**")
        st.write("• Firma tem poder de mercado")
        st.write("• P > RMg = CMg no ótimo")
        st.write("• Ineficiência alocativa")
        
        # Calcular deadweight loss
        competitive_q = (demand_intercept - variable_cost_linear) / (demand_slope + 2 * variable_cost_quadratic)
        if competitive_q > 0:
            dwl = 0.5 * demand_slope * (competitive_q - optimal_quantity) ** 2
            st.write(f"• Perda de bem-estar: R$ {dwl:.2f}")

# Seção educacional
st.markdown("---")
st.subheader("📚 Conceitos Econômicos")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Funções de Custo", "Funções de Receita", "Maximização do Lucro", "Estruturas de Mercado", "Shutdown Point"])

with tab1:
    st.markdown("""
    ### Funções de Custo
    
    **Custo Total (CT):** Soma de todos os custos de produção
    $$CT = CF + CV = CF + c_1 \cdot Q + c_2 \cdot Q^2$$
    
    **Custo Fixo (CF):** Custos que não variam com a produção (ex: aluguel, equipamentos)
    
    **Custo Variável (CV):** Custos que variam com a quantidade produzida (ex: matéria-prima, mão-de-obra)
    
    **Custo Marginal (CMg):** Custo adicional de produzir uma unidade extra
    $$CMg = \\frac{dCT}{dQ} = c_1 + 2c_2 \cdot Q$$
    
    **Custo Médio (CMe):** Custo por unidade produzida
    $$CMe = \\frac{CT}{Q}$$
    
    **Custo Variável Médio (CVMe):** Custo variável por unidade
    $$CVMe = \\frac{CV}{Q}$$
    """)

with tab2:
    st.markdown("""
    ### Funções de Receita
    
    **Receita Total (RT):** Valor total obtido com as vendas
    $$RT = P \cdot Q$$
    
    **Receita Marginal (RMg):** Receita adicional de vender uma unidade extra
    $$RMg = \\frac{dRT}{dQ}$$
    
    #### Concorrência Perfeita:
    - Preço constante: $P = constante$
    - $RT = P \cdot Q$
    - $RMg = P$
    
    #### Monopólio:
    - Função de demanda: $P = a - b \cdot Q$
    - $RT = P \cdot Q = (a - b \cdot Q) \cdot Q$
    - $RMg = a - 2b \cdot Q$
    """)

with tab3:
    st.markdown("""
    ### Maximização do Lucro
    
    **Lucro (π):** Diferença entre receita total e custo total
    $$\pi = RT - CT$$
    
    **Condição de Primeira Ordem:** Para maximizar o lucro
    $$\\frac{d\pi}{dQ} = \\frac{dRT}{dQ} - \\frac{dCT}{dQ} = 0$$
    
    Portanto: **RMg = CMg**
    
    **Condição de Segunda Ordem:** Para garantir máximo
    $$\\frac{d^2\pi}{dQ^2} < 0$$
    
    Ou seja: $\\frac{dCMg}{dQ} > \\frac{dRMg}{dQ}$
    
    **Interpretação:** A firma deve produzir até o ponto onde o custo de produzir uma unidade adicional iguala a receita obtida com essa unidade.
    """)

with tab4:
    st.markdown("""
    ### Estruturas de Mercado
    
    #### Concorrência Perfeita
    - **Características:** Muitos vendedores, produto homogêneo, livre entrada/saída
    - **Poder de mercado:** Nenhum (tomadora de preços)
    - **Eficiência:** Alocativamente eficiente (P = CMg)
    - **Lucro longo prazo:** Zero (devido à livre entrada)
    
    #### Monopólio
    - **Características:** Único vendedor, produto sem substitutos próximos, barreiras à entrada
    - **Poder de mercado:** Total (formadora de preços)
    - **Eficiência:** Ineficiente (P > CMg)
    - **Lucro longo prazo:** Pode ser positivo
    - **Perda de bem-estar:** Deadweight loss devido ao preço acima do custo marginal
    
    **Comparação:** Em monopólio, a quantidade produzida é menor e o preço é maior que em concorrência perfeita, resultando em perda de eficiência econômica.
    """)

with tab5:
    st.markdown("""
    ### Shutdown Point (Ponto de Parada)
    
    **Conceito:** Ponto em que uma firma deve parar a produção no curto prazo
    
    **Condição de Shutdown:**
    $$P < CVMe$$
    
    Onde:
    - P = Preço de venda
    - CVMe = Custo Variável Médio
    
    #### Lógica Econômica:
    
    **Se P < CVMe:**
    - A firma não consegue cobrir nem os custos variáveis
    - Cada unidade produzida gera prejuízo adicional
    - **Decisão:** PARAR a produção (Q = 0)
    - **Prejuízo:** Apenas os custos fixos (CF)
    
    **Se P ≥ CVMe:**
    - A firma cobre os custos variáveis (pelo menos parcialmente)
    - Contribui para cobrir os custos fixos
    - **Decisão:** CONTINUAR produzindo
    - **Mesmo com prejuízo:** Melhor que parar completamente
    
    #### Exemplo Prático:
    Imagine um restaurante:
    - **Custos Fixos:** Aluguel, equipamentos = R$ 5.000/mês
    - **Custos Variáveis:** Ingredientes, funcionários = R$ 15/prato
    - **Preço atual:** R$ 12/prato
    
    Como P (R$ 12) < CVMe (R$ 15), cada prato vendido gera R$ 3 de prejuízo adicional!
    
    **Melhor decisão:** Fechar temporariamente e pagar apenas o aluguel.
    
    #### No Simulador:
    Experimente cenários com custos altos e preços baixos para ver o shutdown point em ação!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>Simulador Microeconômico - Teoria da Firma | Desenvolvido para fins educacionais</small>
</div>
""", unsafe_allow_html=True)
