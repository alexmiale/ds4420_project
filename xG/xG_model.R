library(tidyverse)

shots <- read_csv("shots_clean.csv")
shots2 <- read_csv("shots_clean2.csv")
shots3 <- read_csv("shots_clean3.csv")
shots4 <- read_csv("shots_clean4.csv")
shots <- rbind(shots, shots2, shots3)

# Prepare data
shots$phase_type <- factor(shots$phase_type, 
                           levels = c("finish", "set_play", "transition", "quick_break"))
continuous_cols <- c("distance", "angle", "gk_dist", "defenders_cone", "pressure")
shots[continuous_cols] <- scale(shots[continuous_cols])

X <- model.matrix(~ distance + angle + gk_dist + defenders_cone + pressure +
                    one_touch + is_header + phase_type, data = shots)
y <- shots$goal

# Priors
prior_mu <- matrix(c(0,   # intercept
                     -.7,   # distance
                     .8,   # angle
                     0,   # gk_dist
                     0,   # defenders_cone
                     -.5,   # pressure
                     0,   # one_touch
                     -.4,   # is_header
                     0,   # phase_type set_play
                     0,   # phase_type transition
                     0    # phase_type quick_break
), ncol = 1)

prior_Sigma <- diag(c(
                      8,   # intercept
                      4,   # distance       
                      4,   # angle          
                      8,   # gk_dist
                      8,   # defenders_cone
                      8,   # pressure
                      8,   # one_touch
                      8,   # is_header      
                      8,   # phase_type set_play
                      8,   # phase_type transition
                      8    # phase_type quick_break
))

log_prior <- function(beta) {
  mu  <- as.numeric(prior_mu)
  Sigma <- as.numeric(sqrt(diag(prior_Sigma)))
  sum(dnorm(as.numeric(beta), mean = mu, sd = Sigma, log = TRUE))
}


log_likelihood <- function(beta, X, y) {
    p <- 1 / (1 + exp(-X %*% beta))
    log_lik <- sum(dbinom(y, size = 1, prob = p, log = TRUE))
    return(log_lik)
}

log_posterior <- function(beta, X, y) {
  log_likelihood(beta, X, y) + log_prior(beta)
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
    alpha <- exp(log_posterior(beta_prop, X, y) - log_posterior(beta, X, y))
    
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

samples <- metropolis_sampling(n = 500000, sd = 0.1, burn = 0.18)
print(nrow(samples))
plot(samples[, 1], type = 'l', main = 'Convergence of Y1 Samples', 
     xlab = 'Iteration', ylab = 'Y1')

plot(samples[, 2], type = 'l', main = 'Convergence of Y2 Samples',
     xlab = 'Iteration', ylab = 'Y2')
