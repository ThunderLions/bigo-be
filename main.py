from multiprocessing import Process
import psutil
import time
import matplotlib.pyplot as plt
import string
import random


def kmp_search():
    m = len(pat)
    n = len(txt)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0] * m
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    compute_lps_array(pat, m, lps)

    i = 0  # index for txt[]
    while i < n:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == m:
            print("Found pattern at index " + str(i - j))
            j = lps[j - 1]

        # mismatch after j matches
        elif i < n and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1


def compute_lps_array(pat, M, lps):
    len = 0  # length of the previous longest prefix suffix

    lps[0]  # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len - 1]

                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1


# pat  -> pattern
# txt  -> text
# q    -> A prime number
def rabi_search():
    d = 256
    q = 101

    M = len(pat)
    N = len(txt)
    i = 0
    j = 0
    p = 0  # hash value for pattern
    t = 0  # hash value for txt
    h = 1

    # The value of h would be "pow(d, M-1)%q"
    for i in range(M - 1):
        h = (h * d) % q

    # Calculate the hash value of pattern and first window
    # of text
    for i in range(M):
        p = (d * p + ord(pat[i])) % q
        t = (d * t + ord(txt[i])) % q

    # Slide the pattern over text one by one
    for i in range(N - M + 1):
        # Check the hash values of current window of text and
        # pattern if the hash values match then only check
        # for characters one by one
        if p == t:
            # Check for characters one by one
            for j in range(M):
                if txt[i + j] != pat[j]:
                    break
                else:
                    j += 1

            # if p == t and pat[0...M-1] = txt[i, i+1, ...i+M-1]
            if j == M:
                print("Pattern found at index " + str(i))

        # Calculate hash value for next window of text: Remove
        # leading digit, add trailing digit
        if i < N - M:
            t = (d * (t - ord(txt[i]) * h) + ord(txt[i + M])) % q

            # We might get negative values of t, converting it to
            # positive
            if t < 0:
                t = t + q


def usage_memory_cpu(pid, memory_usage, cpu_usage, pid2=0, sleep_time=0.0005):
    disk_usage = []
    mem_percent = 0
    while pid and psutil.pid_exists(pid) or pid2 and psutil.pid_exists(pid2):
        time.sleep(sleep_time)
        try:
            process = psutil.Process(pid)
            mem = process.memory_info()
            cpu = process.cpu_times()
            disk_usage.append(psutil.disk_usage('/'))
            mem_percent = process.memory_percent()
            memory_usage.append(mem)
            cpu_usage.append(cpu)
        except:
            break

    return [cpu_usage, memory_usage, mem_percent, disk_usage]


def pie_chart(cpu, memory, hard_drive, rss, vms, page_faults):
    pies = [cpu, memory, hard_drive, rss, vms, page_faults]
    labels = ["CPU_usage", "Memory_usage", "Hard_drive_usage", "RSS", "VMS", "Number of Page Faults"]

    # Creating color parameters
    colors = ("lightcoral", "goldenrod", "limegreen", "midnightblue", "plum", "peru")

    # Wedge properties
    wp = {'linewidth': 1, 'edgecolor': "green"}

    # Creating plot
    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, texts, autotexts = ax.pie(pies, autopct='%2.1f%%',
                                      labels=labels,
                                      colors=colors,
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    # Adding legend
    ax.legend(wedges, labels,
              title="System information",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("System usage information")

    # show plot
    plt.show()


def lineChart(title, x1, x2, x3, y1, y2, y3, line1, line2, line3, iter):
    ax1 = fig.add_subplot(2, 3, iter)
    ax1.plot(x1, y1, label=line1)
    ax1.plot(x2, y2, label=line2, linestyle="dashdot", linewidth=1.5, color='mediumslateblue')
    ax1.plot(x3, y3, label=line3, linestyle="--", linewidth=1.5, color='maroon')
    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(title)
    ax1.legend()


def convert_bytes(size):
    return str(round(size / 2. ** 20, 2)) + " MB"


def profilePrinter(cpu, memory, hard_drive, rss, vms, page_faults):
    print("cpu,", cpu)
    print("memory percent,", memory)
    print("hard_drive,", convert_bytes(hard_drive))
    print("rss,", convert_bytes(rss))
    print("vms,", convert_bytes(vms))
    print("page_faults,", page_faults)


def getCordinates(usage_object, start, end):
    l = len(usage_object)
    k = int(end - start // l)
    time_a = []
    for i in range(l):
        time_a.append(i)
        i += k

    return time_a


def keyLineChartForMemory(key, title, t1, t2, t3, it):
    rss1 = map(lambda x: x[key], memory_1)
    rss2 = map(lambda x: x[key], memory_2)
    rss3 = map(lambda x: x[key], memory_3)

    lineChart(title, t1, t2, t3, list(rss1), list(rss2), list(rss3), "KMP", "Rabin karp", "KMP+ Rabin karp", it)


def keyLineChartForCPU(key, title, t1, t2, t3, it):
    rss1 = map(lambda x: x[key], cpu_1)
    rss2 = map(lambda x: x[key], cpu_2)
    rss3 = map(lambda x: x[key], cpu_3)

    lineChart(title, t1, t2, t3, list(rss1), list(rss2), list(rss3), "KMP", "Rabin karp", "KMP+ Rabin karp", it)


def keyLineChartForHardDisk(key, title, t1, t2, t3, it):
    rss1 = map(lambda x: x[key] / 2. ** 15, disk_1)
    rss2 = map(lambda x: x[key] / 2. ** 15, disk_2)
    rss3 = map(lambda x: x[key] / 2. ** 15, disk_3)

    lineChart(title, t1, t2, t3, list(rss1), list(rss2), list(rss3), "KMP", "Rabin karp", "KMP+ Rabin karp", it)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # N is length of the text to searched for matching, the pattern is auto set to 10% of N
    N = 200

    # Initiate the text and pattern to be matched inside the text
    txt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
    pat = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N // 10))
    fig = plt.figure()

    # KMP algo Profiling
    mems = []
    cpus = []
    start_1 = time.time()
    psutil.cpu_percent(interval=None)
    p1 = Process(target=kmp_search())
    p1.start()
    [cpu_1, memory_1, memory_percent_1, disk_1] = usage_memory_cpu(p1.pid, mems, cpus)
    cpu_p_1 = psutil.cpu_percent(interval=1)
    p1.join()
    end_1 = time.time()
    tx1 = getCordinates(cpu_1, start_1, end_1)

    # Rabin karp Algo Profiling
    mems = []
    cpus = []
    start_2 = time.time()
    psutil.cpu_percent(interval=None)
    p2 = Process(target=rabi_search())
    p2.start()
    [cpu_2, memory_2, memory_percent_2, disk_2] = usage_memory_cpu(p2.pid, mems, cpus)
    cpu_p_2 = psutil.cpu_percent(interval=1)
    p2.join()
    end_2 = time.time()
    tx2 = getCordinates(cpu_2, start_2, end_2)

    # Rabin karp + KMP Algo Profiling (Concurrent Process)
    mems = []
    cpus = []
    start_3 = time.time()
    psutil.cpu_percent(interval=None)
    p1 = Process(target=kmp_search())
    p2 = Process(target=rabi_search())
    p1.start()
    p2.start()
    [cpu_3, memory_3, memory_percent_3, disk_3] = usage_memory_cpu(p1.pid, mems, cpus, p2.pid)
    cpu_p_3 = psutil.cpu_percent(interval=1)
    p1.join()
    p2.join()
    end_3 = time.time()
    tx3 = getCordinates(cpu_3, start_3, end_3)

    keyLineChartForMemory(0, "Rss", tx1, tx2, tx3, 1)
    keyLineChartForMemory(1, "Vms", tx1, tx2, tx3, 2)
    keyLineChartForMemory(2, "Page Faults", tx1, tx2, tx3, 3)
    keyLineChartForCPU(0, "User Time", tx1, tx2, tx3, 4)
    keyLineChartForCPU(2, "System Time", tx1, tx2, tx3, 5)
    keyLineChartForHardDisk(0, "Hard Disk Usage", tx1, tx2, tx3, 6)
    plt.subplots_adjust(hspace=0.37)
    plt.show()
    title_sub = "Metrics for text length " + str(N)
    fig.suptitle(title_sub)
    fig.savefig("RabinKarpVsKMP.pdf")

    print("\n KMP algo Profile Information")
    profilePrinter(cpu_p_1, memory_percent_1, disk_1[1][0] - disk_1[-1][0], memory_1[-1].rss, memory_1[-1].vms,
                   memory_1[-1].num_page_faults)

    print("\n Rabin Karp algo Profile Information")
    profilePrinter(cpu_p_2, memory_percent_2, disk_2[1][0] - disk_2[-1][0], memory_2[-1].rss, memory_2[-1].vms,
                   memory_2[-1].num_page_faults)

    print("\n KMP + Rabin Karp algo Profile Information")
    profilePrinter(cpu_p_3, memory_percent_3, disk_3[1][0] - disk_3[-1][0], memory_3[-1].rss, memory_3[-1].vms,
                   memory_3[-1].num_page_faults)
