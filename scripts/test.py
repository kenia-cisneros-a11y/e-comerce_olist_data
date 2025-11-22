import pandas as pd

def tasa_conversion_por_metodo_pago(orders_df, order_payments_df):
    """
    Calcula la tasa de conversión por método de pago.
    ❓ Pregunta de negocio: ¿Qué métodos de pago convierten mejor en ventas?
    📊 Análisis: Permite identificar qué medios de pago generan más órdenes exitosas
    en relación con el total de órdenes. Esto ayuda a priorizar métodos de pago
    más efectivos y detectar aquellos con baja aceptación.
    """
    pagos_por_metodo = order_payments_df.groupby('payment_type')['order_id'].nunique()
    total_ordenes = orders_df['order_id'].nunique()
    tasa_conversion = (pagos_por_metodo / total_ordenes).reset_index()
    tasa_conversion.columns = ['payment_type', 'conversion_rate']

    resumen = order_payments_df.groupby('payment_type')['payment_value'].sum().reset_index()
    resumen.columns = ['payment_type', 'total_payment_value']
    tasa_conversion = tasa_conversion.merge(resumen, on='payment_type', how='left')
    return resumen.sort_values(by='total_payment_value', ascending=False)

def analisis_chargebacks_por_categoria(order_items_df, products_df, order_payments_df):
    """
    Analiza los chargebacks (pagos negativos) por categoría de producto.
    ❓ Pregunta de negocio: ¿Qué categorías de productos generan más devoluciones o fraudes?
    📊 Análisis: Permite detectar categorías con mayor riesgo financiero y ajustar
    políticas de control, devoluciones o prevención de fraude.
    """
    pagos_negativos = order_payments_df[order_payments_df['payment_value'] < 0]
    items_con_chargeback = order_items_df[order_items_df['order_id'].isin(pagos_negativos['order_id'])]
    items_con_categoria = items_con_chargeback.merge(products_df, on='product_id', how='left')
    resumen = items_con_categoria['product_category_name'].value_counts().reset_index()
    resumen.columns = ['product_category_name', 'chargeback_count']
    return resumen

def valor_promedio_vs_cuotas(order_payments_df):
    """
    Calcula el valor promedio de transacción según el número de cuotas.
    ❓ Pregunta de negocio: ¿Cómo afecta el número de cuotas al valor promedio de compra?
    📊 Análisis: Permite entender si los clientes que pagan en más cuotas tienden
    a gastar más o menos, lo que ayuda a diseñar estrategias de financiamiento.
    """
    resumen = order_payments_df.groupby('payment_installments')['payment_value'].mean().reset_index()
    resumen.columns = ['payment_installments', 'avg_transaction_value']
    return resumen.sort_values(by='payment_installments')

def analisis_tiempos_entrega(order_items_df, orders_df):
    """
    Calcula el tiempo promedio de entrega por producto.
    ❓ Pregunta de negocio: ¿Cuánto tardan en promedio los productos en ser entregados?
    📊 Análisis: Permite identificar productos con tiempos de entrega más largos,
    optimizar la logística y mejorar la satisfacción del cliente.
    """
    merged_df = order_items_df.merge(orders_df[['order_id', 'order_purchase_timestamp']], on='order_id', how='left')
    merged_df['delivery_time'] = (merged_df['shipping_limit_date'] - merged_df['order_purchase_timestamp']).dt.days
    resumen = merged_df.groupby('product_id')['delivery_time'].mean().reset_index()
    resumen.columns = ['product_id', 'avg_delivery_time_days']
    return resumen.sort_values(by='avg_delivery_time_days')

def analisis_reembolsos(order_payments_df, orders_df):
    """
    Analiza los reembolsos según el estado de la orden.
    ❓ Pregunta de negocio: ¿En qué estados de orden ocurren más reembolsos?
    📊 Análisis: Ayuda a detectar problemas en el proceso de compra, entrega o
    satisfacción del cliente que derivan en devoluciones.
    """
    reembolsos = order_payments_df[order_payments_df['payment_value'] < 0]
    reembolsos_con_orden = reembolsos.merge(orders_df[['order_id', 'order_status']], on='order_id', how='left')
    resumen = reembolsos_con_orden['order_status'].value_counts().reset_index()
    resumen.columns = ['order_status', 'refund_count']
    return resumen.sort_values(by='refund_count', ascending=False)

def analisis_clientes_frecuentes(orders_df, customers_df):
    """
    Identifica clientes frecuentes con más de 5 órdenes.
    ❓ Pregunta de negocio: ¿Quiénes son los clientes más valiosos por recurrencia?
    📊 Análisis: Permite segmentar clientes VIP y diseñar estrategias de fidelización
    y recompensas para aumentar su lealtad.
    """
    ordenes_por_cliente = orders_df.groupby('customer_id')['order_id'].nunique().reset_index()
    ordenes_por_cliente.columns = ['customer_id', 'total_orders']
    clientes_frecuentes = ordenes_por_cliente[ordenes_por_cliente['total_orders'] > 5]
    resumen = clientes_frecuentes.merge(customers_df, on='customer_id', how='left')
    return resumen.sort_values(by='total_orders', ascending=False)

def analisis_tendencias_temporales(orders_df):
    """
    Analiza la cantidad de órdenes por mes.
    ❓ Pregunta de negocio: ¿Cómo evolucionan las ventas a lo largo del tiempo?
    📊 Análisis: Permite detectar tendencias estacionales, picos de demanda y
    planificar campañas de marketing en los meses más fuertes.
    """
    orders_df['order_month'] = orders_df['order_purchase_timestamp'].dt.to_period('M')
    resumen = orders_df.groupby('order_month')['order_id'].nunique().reset_index()
    resumen.columns = ['order_month', 'total_orders']
    return resumen.sort_values(by='order_month')

def analisis_satisfaccion_clientes(order_reviews_df):
    """
    Analiza la distribución de reseñas por puntaje.
    ❓ Pregunta de negocio: ¿Cuál es el nivel de satisfacción de los clientes?
    📊 Análisis: Permite medir la calidad del servicio y productos, identificar
    áreas de mejora y correlacionar satisfacción con ventas.
    """
    resumen = order_reviews_df.groupby('review_score')['order_id'].nunique().reset_index()
    resumen.columns = ['review_score', 'total_reviews']
    return resumen.sort_values(by='review_score', ascending=False)

def analisis_efectividad_promociones(orders_df, order_payments_df, order_items_df, sellers_df, products_df):
    """
    Calcula la tasa de efectividad de promociones (vouchers) y genera análisis detallados.
    Pregunta de negocio: ¿Qué tan efectivas son las promociones para generar ventas y en qué vendedores/categorías funcionan mejor?
    Análisis:
        - Evalúa si las promociones impulsan la conversión global.
        - Identifica qué vendedores aprovechan más las promociones.
        - Muestra qué categorías de productos se benefician más de las promociones.
    
    Retorna:
        - tasa_efectividad (float): proporción de órdenes con promoción sobre el total.
        - promociones_por_vendedor (DataFrame): número de órdenes con promoción y tasa de efectividad por vendedor.
        - promociones_por_categoria (DataFrame): número de órdenes con promoción y tasa de efectividad por categoría de producto.
    """
    # Filtrar pagos con promociones
    promociones = order_payments_df[order_payments_df['payment_type'] == 'voucher']
    ordenes_con_promocion = orders_df[orders_df['order_id'].isin(promociones['order_id'])]

    # Tasa de efectividad global
    tasa_efectividad = len(ordenes_con_promocion) / len(orders_df)

    # --- Promociones por vendedor ---
    items_promocion = order_items_df[order_items_df['order_id'].isin(promociones['order_id'])]
    items_vendedor = order_items_df.merge(sellers_df, on='seller_id', how='left')

    # Total de órdenes por vendedor
    total_por_vendedor = items_vendedor.groupby('seller_id')['order_id'].nunique().reset_index()
    total_por_vendedor.columns = ['seller_id', 'total_orders']

    # Órdenes con promoción por vendedor
    promo_por_vendedor = items_promocion.merge(sellers_df, on='seller_id', how='left')
    promo_por_vendedor = promo_por_vendedor.groupby('seller_id')['order_id'].nunique().reset_index()
    promo_por_vendedor.columns = ['seller_id', 'promo_orders_count']

    # Merge y cálculo de tasa
    promociones_por_vendedor = promo_por_vendedor.merge(total_por_vendedor, on='seller_id', how='left')
    promociones_por_vendedor['promo_effectiveness_rate'] = promociones_por_vendedor['promo_orders_count'] / promociones_por_vendedor['total_orders']
    promociones_por_vendedor = promociones_por_vendedor.sort_values(by='promo_effectiveness_rate', ascending=False)

    # --- Promociones por categoría ---
    items_categoria = order_items_df.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')

    # Total de órdenes por categoría
    total_por_categoria = items_categoria.groupby('product_category_name')['order_id'].nunique().reset_index()
    total_por_categoria.columns = ['product_category_name', 'total_orders']

    # Órdenes con promoción por categoría
    promo_por_categoria = items_promocion.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    promo_por_categoria = promo_por_categoria.groupby('product_category_name')['order_id'].nunique().reset_index()
    promo_por_categoria.columns = ['product_category_name', 'promo_orders_count']

    # Merge y cálculo de tasa
    promociones_por_categoria = promo_por_categoria.merge(total_por_categoria, on='product_category_name', how='left')
    promociones_por_categoria['promo_effectiveness_rate'] = promociones_por_categoria['promo_orders_count'] / promociones_por_categoria['total_orders']
    promociones_por_categoria = promociones_por_categoria.sort_values(by='promo_effectiveness_rate', ascending=False)

    return tasa_efectividad, promociones_por_vendedor, promociones_por_categoria


def analisis_ventas_cruzadas(order_items_df, products_df):
    """
    Analiza la diversidad de categorías compradas por orden.
    ❓ Pregunta de negocio: ¿Cuántas categorías diferentes compran los clientes en una sola orden?
    📊 Análisis: Permite medir el potencial de ventas cruzadas y diseñar estrategias
    de bundles o recomendaciones de productos.
    """
    merged_df = order_items_df.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    categorias_por_orden = merged_df.groupby('order_id')['product_category_name'].nunique().reset_index()
    categorias_por_orden.columns = ['order_id', 'num_categories']
    resumen = categorias_por_orden['num_categories'].value_counts().reset_index()
    resumen.columns = ['num_categories', 'total_orders']
    return resumen.sort_values(by='num_categories')

def analisis_impacto_ubicacion(customers_df, orders_df):
    """
    Analiza el impacto de la ubicación geográfica en las órdenes.
    ❓ Pregunta de negocio: ¿Qué estados generan más ventas?
    📊 Análisis: Permite identificar regiones estratégicas, optimizar campañas
    locales y ajustar la logística según la demanda.
    """
    merged_df = orders_df.merge(customers_df[['customer_id', 'customer_state']], on='customer_id', how='left')
    resumen = merged_df.groupby('customer_state')['order_id'].nunique().reset_index()
    resumen.columns = ['customer_state', 'total_orders']
    return resumen.sort_values(by='total_orders', ascending=False)

def analisis_impacto_logistica(order_items_df, orders_df):
    """
    Analiza el impacto de la logística según el estado de las órdenes.
    ❓ Pregunta de negocio: ¿Cómo afecta el estado de la orden a la cantidad de órdenes procesadas?
    📊 Análisis: Permite identificar cuellos de botella en la logística y mejorar 
    procesos para aumentar la eficiencia operativa.
    """
    merged_df = order_items_df.merge(orders_df[['order_id', 'order_status']], on='order_id', how='left')
    resumen = merged_df.groupby('order_status')['order_id'].nunique().reset_index()
    resumen.columns = ['order_status', 'total_orders']
    return resumen.sort_values(by='total_orders', ascending=False)

