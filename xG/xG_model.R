library(tidyverse)
library(ggplot2)
library(rsample)
library(ggsoccer)
library(Metrics)
library(pROC)
set.seed(07)

shots <- read_csv('../data/shots_clean.csv')
shots <- rbind(shots)


# Prepare data
shots$phase_type <- factor(shots$phase_type, 
                           levels = c('finish', 'set_play', 'transition', 'quick_break'))
continuous_cols <- c('distance', 'angle', 'gk_dist', 'gk_angle_blocked', 'defenders_cone', 'pressure')
shots[continuous_cols] <- scale(shots[continuous_cols])
head(shots)

data_split <- initial_split(shots, prop = 0.75, strata = 'goal')
train_data <- training(data_split)
test_data  <- testing(data_split)


X <- model.matrix(~ distance + angle + gk_dist + gk_angle_blocked + defenders_cone + pressure +
                    one_touch + is_header + phase_type, data = train_data)
y <- train_data$goal

# Priors
prior_mu <- matrix(c(-1,     # intercept
                     -2,     # distance
                      2,     # angle
                      .2,    # gk_dist
                       -1,   # gk_angle_blocked
                      -.3,   # defenders_cone
                      -.5,   # pressure
                      .3,    # one_touch
                      -.3,   # is_header
                      -.1,   # phase_type set_play
                      .3,    # phase_type transition
                      0      # phase_type quick_break
), ncol = 1)

prior_Sigma <- diag(c(1,  # intercept
                      1, # distance       
                      1, # angle          
                      1,   # gk_dist
                      1,   # gk_angle_blocked
                      1,   # defenders_cone
                      1,   # pressure
                      1,   # one_touch
                      1,   # is_header      
                      1,   # phase_type set_play
                      1,   # phase_type transition
                      1    # phase_type quick_break
))

prior <- function(beta) {
  mu  <- as.numeric(prior_mu)
  sds <- as.numeric(sqrt(diag(prior_Sigma)))
  prod(dnorm(as.numeric(beta), mean = mu, sd = sds))
}

likelihood <- function(beta, X, y) {
  p <- 1 / (1 + exp(-X %*% beta))
  prod(dbinom(as.vector(y), size = 1, prob = p))
}

posterior <- function(beta, X, y) {
  likelihood(beta, X, y) * prior(beta)
}

metropolis_sampling <- function(n = 10000, sd = 1.25, burn = 0.18) {
  
  samples <- list()
  n_params <- ncol(X)
  total_samples <- floor(n / (1 - burn))
  m <- as.integer(total_samples * burn)
  accepted <- 0
  
  beta <- rep(0, n_params)
  
  while (length(samples) < total_samples) {
    beta_prop <- rnorm(n_params, beta, sd)
    alpha <- posterior(beta_prop, X, y) / posterior(beta, X, y)
    
    if (runif(1) < alpha) {
      beta <- beta_prop
      accepted <- accepted + 1
    }
    samples[[length(samples) + 1]] <- beta
  }
  
  acceptance_rate <- accepted / total_samples
  cat(sprintf('Acceptance rate: %.3f\n', acceptance_rate))
  
  samples <- do.call(rbind, samples)
  samples <- samples[(m + 1):nrow(samples), ]
  
  return(samples)
}


samples <- metropolis_sampling(n = 500000, sd = 0.22, burn = 0.18)
print(nrow(samples))
png('visualizations/convergence_y1.png', width = 800, height = 600)
plot(samples[, 1], type = 'l', main = 'Convergence of Y1 Samples', 
     xlab = 'Iteration', ylab = 'Y1')
dev.off()

png('visualizations/convergence_y2.png', width = 800, height = 600)
plot(samples[, 2], type = 'l', main = 'Convergence of Y2 Samples',
     xlab = 'Iteration', ylab = 'Y2')
dev.off()

png('visualizations/acf_y1.png', width = 800, height = 600)
acf(samples[, 1], main = 'Metropolis Y1 Samples ACF Plot')
dev.off()

png('visualizations/acf_y2.png', width = 800, height = 600)
acf(samples[, 2], main = 'Metropolis Y2 Samples ACF Plot')
dev.off()

thinned_samples <- samples[seq(1, nrow(samples), by = 50), ] 

png('visualizations/acf_thinned_y1.png', width = 800, height = 600)
acf(thinned_samples[, 1], main = 'Metropolis Y1 (thinned) Samples ACF Plot')
dev.off()

png('visualizations/acf_thinned_y2.png', width = 800, height = 600)
acf(thinned_samples[, 2], main = 'Metropolis Y2 (thinned) Samples ACF Plot')
dev.off()

colMeans(thinned_samples)



# Predict Test Data
X_test <- model.matrix(~ distance + angle + gk_dist + gk_angle_blocked + defenders_cone + pressure +
                         one_touch + is_header + phase_type, data = test_data)
y_test <- test_data$goal

sigmoid <- function(x) 1 / (1 + exp(-x))
pred <- sigmoid(thinned_samples %*% t(X_test))

results_df <- data.frame(
  pred = colMeans(pred),
  actual = factor(y_test, levels = c(0, 1))
)
head(results_df, 20)

# Plot all shots
results_df$ball_x <- test_data$ball_x 
results_df$ball_y <- test_data$ball_y - 34
results_df$is_header <- factor(test_data$is_header, levels = c(0, 1))

pitch_custom <- list(
  length = 105,
  width = 68,
  penalty_box_length = 16.5,
  penalty_box_width = 40.32,
  six_yard_box_length = 5.5,
  six_yard_box_width = 18.32,
  penalty_spot_distance = 11,
  goal_width = 7.32,
  origin_x = -50,
  origin_y = -75
)

ggplot(results_df) +
  annotate_pitch(dimensions = pitch_custom) +
  geom_point(aes(x=ball_x, y=ball_y, size=pred, color=is_header, shape = actual)) +
  geom_text(data = results_df,
            aes(x = ball_x, y = ball_y, label = round(pred, 2)),
            color = '#111111', size = 3, vjust = -1.5) +
  scale_size_continuous(name = 'xG') +
  scale_color_manual(name = 'Header', values = c('0' = '#355070', '1' = '#D90429')) +
  scale_shape_manual(name = 'Outcome', values = c('0' = 1, '1' = 16)) +
  labs(title = 'Goals — predicted xG') +
  theme_pitch() +
  theme(plot.title = element_text(color = '#111111', hjust = 0.5, size = 14))


# Plot Distribution of a single shot
for (i in 10:15) { 
  shot_xg_dist <- pred[, i]
  
  png(paste0('visualizations/histogram_', i, '.png'), width = 800, height = 600)
  hist_plot <- hist(shot_xg_dist, 
       breaks = 50,
       main   = paste('xG distribution — shot', i),
       col    = '#355070',
       border = 'white')
  dev.off()
  
  shot_plot <- ggplot(results_df) +
    annotate_pitch(dimensions = pitch_custom) +
    geom_point(data = results_df[i,],
               aes(x=ball_x, y=ball_y, size=pred, color=is_header, shape = actual)) +
    geom_text(data = results_df[i,],
              aes(x = ball_x, y = ball_y, label = round(pred, 2)),
              color = '#111111', size = 3, vjust = -1.5) +
    scale_size_continuous(name = 'xG') +
    scale_color_manual(name = 'Header', values = c('0' = '#355070', '1' = '#D90429')) +
    scale_shape_manual(name = 'Outcome', values = c('0' = 1, '1' = 16)) +
    labs(title = 'Goals — predicted xG') +
    theme_pitch() +
    theme(plot.title = element_text(color = '#111111', hjust = 0.5, size = 14))
  
    ggsave(paste0('visualizations/shot_plot_', i, '.png'), plot=shot_plot, dpi = 150)
}

# Evaluating the model
results_df$actual <- as.numeric(as.character(results_df$actual))
results_df$pred   <- as.numeric(as.character(results_df$pred))

log_loss <- logLoss(results_df$actual, results_df$pred)
auc_score  <- auc(roc(results_df$actual, results_df$pred))
cat('Log Loss: ', round(log_loss,  4), '\n')
cat('AUC Score: ', round(auc_score, 4), '\n')
