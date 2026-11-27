"""Genera el diagrama CRISP-DM vectorial y PNG para el informe."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    """Dibuja el flujo real y separa validación de prueba final."""
    out = Path(sys.argv[1] if len(sys.argv)>1 else 'docs'); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(10,7.4)); ax.set(xlim=(0,100),ylim=(0,74)); ax.axis('off'); fig.subplots_adjust(0,0,1,1)
    navy='#193B53'; teal='#176B67'; gray='#52616B'
    def box(x,y,w,h,title,lines):
        """Renderiza una fase con sus acciones y evidencia."""
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.25,rounding_size=.8',fc='#F4F8FA',ec=navy,lw=1.2))
        ax.text(x+1.5,y+h-2,title,fontsize=11,weight='bold',color=navy,va='top')
        ax.text(x+1.5,y+h-6,'\n'.join(lines),fontsize=9.3,color='#263746',va='top',linespacing=1.45)
    def arrow(a,b,c=navy,style='-'):
        """Conecta fases o representa retroalimentación."""
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=c,lw=1.4,linestyle=style))
    ax.text(3,71,'CRISP-DM | Calidad y tamaño de frutas',fontsize=18,weight='bold',color=navy,va='top')
    ax.text(3,66.8,'Flujo del proyecto • seis especies • tres clases de calidad',fontsize=10,color=gray)
    box(3,45,44,18,'01  Comprensión del negocio',['Clasificación manual lenta y subjetiva','Calidad: Buena / Regular / Mala','Tamaño relativo; apoyo a decisiones'])
    box(53,45,44,18,'02  Comprensión de datos',['Kaggle: 19 526 imágenes; 6 especies','Desbalance y atajo del fondo','Mixed: sin etiquetas individuales'])
    box(53,24,44,18,'03  Preparación de datos',['Mixed: 760 recortes anotados','Grupos por objeto físico inferido','Train / validación / prueba por grupo'])
    box(3,24,44,18,'04  Modelado',['Regresión logística, SVM y RF','CNN desde cero: T6 y T7','Ajuste agrupado; control solo borde'])
    box(3,3,44,18,'05  Evaluación',['Validación: T6, F1 macro 0.725','Prueba final: Kaggle + Mixed y','69 fotos propias; T6 F1 macro 0.439'])
    box(53,3,44,18,'06  Despliegue',['Streamlit: imagen o cámara','CNN T6: calidad y confianza','Segmentación y longitud normalizada'])
    arrow((47.5,54),(52.5,54)); arrow((75,44.5),(75,42.5)); arrow((52.5,33),(47.5,33)); arrow((25,23.5),(25,21.5)); arrow((47.5,12),(52.5,12))
    ax.text(50,1.2,'Prueba final reservada: los errores orientan trabajo futuro, no reajuste sobre el test.',ha='center',fontsize=8.5,color=teal,weight='bold')
    ax.text(50,64.5,'Ética, impactos y trazabilidad en todas las fases',ha='center',fontsize=8.5,color=teal)
    # Iteración dentro de entrenamiento/validación, antes de la prueba final.
    arrow((40,21.5),(40,23.6),teal,'--')
    ax.text(29,22.5,'Ajuste por val.',fontsize=7.5,color=teal,va='center')
    for ext in ('svg','pdf','png'):
        fig.savefig(out/f'diagrama_crispdm.{ext}',dpi=200 if ext=='png' else None,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    main()
