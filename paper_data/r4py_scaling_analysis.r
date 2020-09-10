#
# Analysis Repast4Py Performance Testing.  
#
#
#

library(data.table)
library(ggplot2)

# Std Err
std <- function(x) sd(x)/sqrt(length(x))

# 95% CI
z <- 1.960

dt <- NULL
table <- NULL

# Load all of the stats files that exist in an experiments dir
files <- list.files (path=".", recursive=FALSE, pattern = "*.csv")

tableList <- list()
for (f in files){
  table <- fread(f)
  
  
  tableList[[f]] <- table
}

dt <- rbindlist(tableList)  # Stack the list of tables into a single DT
tableList <- NULL           # clear mem

setnames(dt, c("proc", "humans", "zombies", "foo", "seed", "run_time"))

# Mean, SD, STD across replicates
stats_summary <- dt[, list(run_time=mean(run_time), run_time.sd=sd(run_time), run_time.std=std(run_time)), 
                          by=list(proc)]


base_run_time <- stats_summary[proc == 36]$run_time

stats_summary[, speedup := base_run_time / run_time]

# The colorblind palette with grey:
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

p <- ggplot(stats_summary) + geom_line(aes(x=proc, y=run_time), size=1, alpha=0.5) +
  geom_point(aes(x=proc, y=run_time), size=2, alpha=0.5) +
  
#  geom_ribbon(aes(x=proc, ymin=run_time-z*run_time.std, ymax=run_time+z*run_time.std),alpha=0.3,colour=NA) +
  
#  geom_ribbon(aes(x=proc, ymin=run_time-run_time.sd, ymax=run_time+run_time.sd),alpha=0.3) +

  geom_errorbar(aes(x=proc, ymin=run_time-3*run_time.sd, ymax=run_time+3*run_time.sd),alpha=0.95,width=0.025) +
  
  scale_x_log10(limits = c(10,1000)) +
  scale_y_log10(limits = c(10,1000)) +
  
#  scale_x_continuous(limits = c(1,12), breaks=seq(1,12,1)) +
#  scale_colour_manual(values=cbPalette) +
#  scale_fill_manual(values=cbPalette) + 
  labs(y="Run Time (s)", x="Procs", title="Zombies Model Runtime Scaling") +
  
  theme_bw() +
  theme(axis.text=element_text(size=14),axis.title=element_text(size=14),legend.text=element_text(size=14))

show(p)
ggsave(p, filename="zombies_scaling.png", width=10, height=8)

q <- ggplot(stats_summary) + 
#  geom_line(aes(x=proc, y=speedup), size=1, alpha=0.5) +
  geom_point(aes(x=proc, y=speedup), size=4, alpha=0.5) +
  
  geom_smooth(aes(x=proc, y=speedup), method='lm', formula= y ~ x, se = FALSE) +
  
  #  scale_x_continuous(limits = c(1,12), breaks=seq(1,12,1)) +
  #  scale_colour_manual(values=cbPalette) +
  #  scale_fill_manual(values=cbPalette) + 
  labs(y="Performance Speedup", x="Procs", title="Zombies Model Runtime Scaling") +
  
  theme_bw() +
  theme(axis.text=element_text(size=14),axis.title=element_text(size=14),legend.text=element_text(size=14))

show(q)
ggsave(p, filename="zombies_speedup.png", width=10, height=8)
