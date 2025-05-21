"""
Speculative decoding implementation for accelerating LLM inference.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class SpeculativeDecoder:
    """
    Implements speculative decoding for faster text generation.
    
    Uses a smaller draft model to propose multiple tokens at once,
    which are then verified by the target model, significantly
    reducing the number of sequential steps needed.
    """
    
    def __init__(self,
                target_model: nn.Module,
                draft_model: nn.Module,
                tokenizer: Any,
                num_speculative_tokens: int = 4,
                max_accept_length: Optional[int] = None,
                temperature: float = 1.0,
                device: str = "cuda"):
        """
        Initialize speculative decoder.
        
        Args:
            target_model: The large, high-quality language model
            draft_model: The smaller, faster language model for drafting
            tokenizer: Tokenizer for processing text
            num_speculative_tokens: Number of tokens to speculatively generate
            max_accept_length: Maximum consecutive tokens to accept without verification
            temperature: Sampling temperature
            device: Device to run models on
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self.max_accept_length = max_accept_length or num_speculative_tokens
        self.temperature = temperature
        self.device = device
        
        # Put models in evaluation mode
        self.target_model.eval()
        self.draft_model.eval()
        
        # Performance metrics
        self.metrics = {
            "total_tokens_generated": 0,
            "draft_tokens_accepted": 0,
            "draft_tokens_rejected": 0,
            "verification_steps": 0,
            "draft_acceptance_rate": 0.0,
            "speedup_factor": 1.0
        }
    
    def generate(self,
               prompt: Union[str, List[int]],
               max_length: int = 100,
               top_p: float = 0.9,
               top_k: int = 0,
               temperature: Optional[float] = None,
               callback: Optional[Callable[[str], None]] = None) -> List[int]:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Text prompt or token IDs
            max_length: Maximum number of tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            temperature: Sampling temperature (overrides instance default)
            callback: Optional callback for streaming generated text
            
        Returns:
            output_tokens: Generated token sequence
        """
        # Use instance temperature if not specified
        temperature = temperature if temperature is not None else self.temperature
        
        # Tokenize prompt if needed
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            input_ids = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Initialize generation
        generated_tokens = input_ids.tolist()[0]
        prompt_length = len(generated_tokens)
        target_length = prompt_length + max_length
        
        # Track metrics
        tokens_from_draft = 0
        draft_calls = 0
        target_calls = 0
        
        # Generate tokens
        with torch.no_grad():
            # Start tracking time
            start_time = time.time()
            
            while len(generated_tokens) < target_length:
                # Get current input for draft model
                current_input_ids = torch.tensor([generated_tokens], dtype=torch.long, device=self.device)
                
                # Draft multiple tokens
                draft_outputs = self._generate_draft_tokens(
                    current_input_ids,
                    num_tokens=self.num_speculative_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                draft_calls += 1
                
                # Extract draft tokens
                draft_tokens = draft_outputs["token_ids"][0].tolist()
                draft_logits = draft_outputs["logits"]
                
                # Verify with target model
                verification_result = self._verify_draft_tokens(
                    current_input_ids,
                    draft_tokens,
                    draft_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                target_calls += 1
                
                # Extract accepted tokens
                accepted_tokens = verification_result["accepted_tokens"]
                next_token = verification_result["next_token"]
                
                # Update metrics
                tokens_from_draft += len(accepted_tokens)
                
                # Add accepted tokens to output
                generated_tokens.extend(accepted_tokens)
                
                # Add the next token (either target model's or a draft accepted token)
                if next_token is not None:
                    generated_tokens.append(next_token)
                
                # Call callback if provided
                if callback is not None:
                    new_tokens = accepted_tokens + ([next_token] if next_token is not None else [])
                    if new_tokens:
                        new_text = self.tokenizer.decode(new_tokens)
                        callback(new_text)
                
                # Break if we've reached target length
                if len(generated_tokens) >= target_length:
                    break
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
        
        # Number of tokens generated (excluding prompt)
        num_generated = len(generated_tokens) - prompt_length
        
        # Update metrics
        self.metrics["total_tokens_generated"] += num_generated
        self.metrics["draft_tokens_accepted"] += tokens_from_draft
        self.metrics["draft_tokens_rejected"] += (draft_calls * self.num_speculative_tokens) - tokens_from_draft
        self.metrics["verification_steps"] += target_calls
        
        # Calculate acceptance rate and speedup
        if draft_calls * self.num_speculative_tokens > 0:
            acceptance_rate = tokens_from_draft / (draft_calls * self.num_speculative_tokens)
            self.metrics["draft_acceptance_rate"] = acceptance_rate
        
        if num_generated > 0 and target_calls > 0:
            self.metrics["speedup_factor"] = num_generated / target_calls
        
        # Calculate tokens per second
        if elapsed_time > 0:
            tokens_per_second = num_generated / elapsed_time
            logger.info(f"Generated {num_generated} tokens in {elapsed_time:.2f}s "
                      f"({tokens_per_second:.2f} tokens/sec, speedup: {self.metrics['speedup_factor']:.2f}x)")
        
        return generated_tokens

    def _generate_draft_tokens(self,
                            input_ids: torch.Tensor,
                            num_tokens: int,
                            temperature: float = 1.0,
                            top_p: float = 1.0,
                            top_k: int = 0) -> Dict[str, Any]:
        """
        Generate draft tokens using the smaller model.
        
        Args:
            input_ids: Input token IDs
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            output: Dictionary with generated tokens and their logits
        """
        # Initialize output tracking
        batch_size = input_ids.shape[0]
        generated_ids = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=self.device)
        logits_sequence = []
        
        # Start with input_ids
        current_ids = input_ids
        
        # Generate tokens autoregressively
        for i in range(num_tokens):
            # Forward pass through draft model
            with torch.no_grad():
                outputs = self.draft_model(current_ids)
                next_token_logits = outputs.logits[:, -1, :]
            
            # Store logits
            logits_sequence.append(next_token_logits)
            
            # Sample next token
            next_token = self._sample_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            # Store token
            generated_ids[:, i] = next_token.squeeze(-1)
            
            # Prepare for next iteration
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return {
            "token_ids": generated_ids,
            "logits": logits_sequence
        }
    
    def _verify_draft_tokens(self,
                          input_ids: torch.Tensor,
                          draft_tokens: List[int],
                          draft_logits: List[torch.Tensor],
                          temperature: float = 1.0,
                          top_p: float = 1.0,
                          top_k: int = 0) -> Dict[str, Any]:
        """
        Verify draft tokens with the target model.
        
        Args:
            input_ids: Original input token IDs
            draft_tokens: Draft tokens to verify
            draft_logits: Logits from the draft model
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            result: Dictionary with accepted tokens and next token
        """
        # Handle empty draft case
        if not draft_tokens:
            # Generate a single token with the target model
            with torch.no_grad():
                target_outputs = self.target_model(input_ids)
                target_logits = target_outputs.logits[:, -1, :]
                
                # Sample next token
                next_token = self._sample_token(
                    target_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                ).item()
            
            return {
                "accepted_tokens": [],
                "next_token": next_token,
                "accept_length": 0
            }
        
        # Concatenate input with draft tokens
        joint_sequence = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], dtype=torch.long, device=self.device)
        ], dim=1)
        
        # Get target model logits for the entire sequence
        with torch.no_grad():
            target_outputs = self.target_model(joint_sequence)
            target_logits_sequence = target_outputs.logits[:, input_ids.shape[1]-1:-1, :]
        
        # Compare target and draft distributions
        accepted_tokens = []
        
        for i, (target_logits, draft_logits_i) in enumerate(zip(target_logits_sequence, draft_logits)):
            # Apply temperature
            if temperature > 0:
                target_probs = F.softmax(target_logits / temperature, dim=-1)
                draft_probs = F.softmax(draft_logits_i / temperature, dim=-1)
            else:
                # Use one-hot for greedy sampling
                target_probs = F.one_hot(torch.argmax(target_logits, dim=-1), num_classes=target_logits.shape[-1]).float()
                draft_probs = F.one_hot(torch.argmax(draft_logits_i, dim=-1), num_classes=draft_logits_i.shape[-1]).float()
            
            # Get the draft token
            draft_token = draft_tokens[i]
            
            # Compute acceptance probability
            # min(1, target_prob / draft_prob) for the selected token
            target_prob = target_probs[0, draft_token].item()
            draft_prob = draft_probs[0, draft_token].item()
            
            # Add small epsilon to avoid division by zero
            accept_prob = min(1.0, target_prob / (draft_prob + 1e-10))
            
            # Make stochastic decision whether to accept
            if torch.rand(1).item() < accept_prob:
                # Accept this token
                accepted_tokens.append(draft_token)
            else:
                # Reject this and all subsequent tokens
                break
        
        # If we accepted all tokens and haven't reached maximum, use the target model's
        # prediction for the next token after the sequence
        next_token = None
        if len(accepted_tokens) == len(draft_tokens) and len(accepted_tokens) < self.max_accept_length:
            # Get token after the last draft token
            last_target_logits = target_outputs.logits[:, -1, :]
            next_token = self._sample_token(
                last_target_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            ).item()
        elif len(accepted_tokens) < len(draft_tokens):
            # If we rejected a token, use the target model's prediction at that position
            next_logits = target_logits_sequence[len(accepted_tokens)]
            next_token = self._sample_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            ).item()
        
        return {
            "accepted_tokens": accepted_tokens,
            "next_token": next_token,
            "accept_length": len(accepted_tokens)
        }
    
    def _sample_token(self,
                    logits: torch.Tensor,
                    temperature: float = 1.0,
                    top_p: float = 1.0,
                    top_k: int = 0) -> torch.Tensor:
        """
        Sample next token from logits.
        
        Args:
            logits: Token logits
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            token_id: Sampled token ID
        """
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # If temperature is 0, use greedy sampling
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return dict(self.metrics)