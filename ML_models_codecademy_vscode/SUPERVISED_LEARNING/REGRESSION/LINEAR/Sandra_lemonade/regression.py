
from matplotlib import pyplot as plt


def get_gradient_at_b(x,y,m,b):
    diff=0
    for i in range(len(y)):
        diff+=(y[i]-(m*x[i]+b))

    b_gradient=(-2/len(y))*diff
    return b_gradient

def get_gradient_at_m(x,y,m,b):
    diff=0
    for i in range(len(y)):
        diff+=x[i]*(y[i]-(m*x[i]+b))

    m_gradient=(-2/len(y))*diff
    return m_gradient

def step_gradient(x,y,b_current,m_current,learning_rate):
    b_gradient=get_gradient_at_b(x,y,m_current,b_current)
    m_gradient=get_gradient_at_m(x,y,m_current,b_current)

    b=b_current-(learning_rate*b_gradient)
    m=m_current-(learning_rate*m_gradient)

    return [b,m]

def gradient_descent(x,y,learning_rate,num_iterations):
    b=m=0
    b_change=[]
    m_change=[]
    for i in range(num_iterations):
        b,m=step_gradient(x,y,b,m,learning_rate)
        b_change.append(b)
        m_change.append(m)
    return (b,m,b_change,m_change)



#--------------FUNCTION CALLING--------------------------------------#

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

b_change_list=[]
m_change_list=[]
lr=0.01
b,m,b_change_list,m_change_list=gradient_descent(months,revenue,lr,1000)
print(b,m)
y=[m*x+b for x in months]
#------plotting results-----------------#

plt.plot(months,revenue,"o")
plt.plot(months,y)
plt.title(f"Best fit line for Sandra\'s Lemonade Stall with intercept {int(b)} & slope {int(m)}")
plt.xlabel("months")
plt.ylabel("revenue")
plt.show()
plt.close()

#----Plotting convergence results-------#
plt.plot(list(range(1000)),b_change_list)
plt.title(f"change of b across iterations with a learning rate of {lr}")
plt.xlabel("ITERATIONS")
plt.ylabel("b")
plt.show()
plt.close()

plt.plot(list(range(1000)),m_change_list)
plt.title(f"change of m across iterations with a learning rate of {lr}")
plt.xlabel("ITERATIONS")
plt.ylabel("m")
plt.show()
plt.close()













