def consume_async_generator(generator):
    """Helper to consume async generator synchronously."""
    import asyncio
    
    text_parts = []
    
    async def _consume():
        chunk_count = 0
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            async for chunk in generator:
                chunk_count += 1
                chunk_type = type(chunk).__name__
                
                # Log first few chunks for debugging
                if chunk_count <= 3:
                    logger.debug(f"Chunk {chunk_count}: type={chunk_type}, value={str(chunk)[:100]}")
                
                # Ultra-safe: only handle strings, convert everything else immediately
                # Don't check types or access any attributes that might trigger Pydantic
                try:
                    # If it's already a string, use it directly
                    if chunk is None:
                        continue
                    elif isinstance(chunk, str):
                        text_parts.append(chunk)
                    else:
                        # For non-strings, try multiple strategies to extract text
                        text_extracted = False
                        
                        # Strategy 1: Try vars() to get __dict__ without triggering properties
                        try:
                            chunk_vars = vars(chunk)
                            if 'text' in chunk_vars:
                                text_val = chunk_vars['text']
                                if isinstance(text_val, str):
                                    text_parts.append(text_val)
                                    text_extracted = True
                                elif text_val is not None:
                                    text_parts.append(str(text_val))
                                    text_extracted = True
                        except (TypeError, AttributeError):
                            pass
                        
                        # Strategy 2: Try object.__getattribute__ to bypass properties
                        if not text_extracted:
                            try:
                                text_val = object.__getattribute__(chunk, 'text')
                                if isinstance(text_val, str):
                                    text_parts.append(text_val)
                                    text_extracted = True
                            except (AttributeError, TypeError):
                                pass
                        
                        # Strategy 3: Try direct string conversion
                        if not text_extracted:
                            try:
                                text_parts.append(str(chunk))
                                text_extracted = True
                            except:
                                pass
                        
                        # Strategy 4: Last resort - use repr
                        if not text_extracted:
                            try:
                                text_parts.append(repr(chunk))
                            except:
                                text_parts.append(f"<{chunk_type}>")
                                
                except AttributeError as ae:
                    # Specifically catch model_copy errors
                    if 'model_copy' in str(ae):
                        # This is the model_copy error - just convert to string
                        try:
                            text_parts.append(str(chunk))
                        except:
                            text_parts.append("")
                    else:
                        raise
                except Exception as e:
                    # Log unexpected errors but continue
                    logger.debug(f"Error processing chunk {chunk_count}: {e}")
                    try:
                        text_parts.append(str(chunk))
                    except:
                        pass
                        
        except AttributeError as ae:
            # Catch model_copy errors at the generator level too
            error_msg = str(ae)
            if 'model_copy' in error_msg:
                # Generator iteration failed due to model_copy
                logger.warning(f"model_copy error during iteration: {ae}")
                result_text_so_far = "".join(text_parts)
                logger.warning(f"Processed {chunk_count} chunks before error, collected {len(result_text_so_far)} chars")
                # Return what we have so far, even if incomplete - might still be usable
                if result_text_so_far:
                    return result_text_so_far
                # If we have nothing, this will trigger the fallback
                raise
            else:
                raise
        except Exception as e:
            # Log any other errors but don't fail completely
            error_msg = str(e)
            if 'model_copy' in error_msg:
                logger.warning(f"model_copy error: {e}, processed {chunk_count} chunks")
                result_text_so_far = "".join(text_parts)
                # Return partial results if we have any
                if result_text_so_far:
                    return result_text_so_far
                # If no partial results, raise to trigger fallback
                raise
            else:
                logger.warning(f"Error consuming generator: {e}, processed {chunk_count} chunks")
                # For non-model_copy errors, raise to surface the issue
                raise
        
        result_text = "".join(text_parts)
        logger.debug(f"Consumed {chunk_count} chunks, total text length: {len(result_text)}")
        
        if not result_text and chunk_count == 0:
            logger.warning("No chunks were processed from the generator - it may have failed to start")
        
        return result_text
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    try:
        if loop.is_running():
            # If we are already in a loop, we can't use run_until_complete
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _consume())
                result_text = future.result()
        else:
            result_text = loop.run_until_complete(_consume())
    except AttributeError as ae:
        # Catch model_copy errors during loop execution
        if 'model_copy' in str(ae):
            # Return any partial text we collected before the error
            import logging
            logging.getLogger(__name__).warning(f"model_copy error in loop execution: {ae}")
            result_text_so_far = "".join(text_parts) if 'text_parts' in locals() else ""
            if result_text_so_far:
                logging.getLogger(__name__).info(f"Returning partial result with {len(result_text_so_far)} chars")
                return result_text_so_far
            # If no partial results, return empty to trigger fallback
            return ""
        raise
    except Exception as e:
        # Catch any other errors
        import logging
        error_msg = str(e)
        if 'model_copy' in error_msg:
            logging.getLogger(__name__).warning(f"model_copy error: {e}")
            result_text_so_far = "".join(text_parts) if 'text_parts' in locals() else ""
            if result_text_so_far:
                logging.getLogger(__name__).info(f"Returning partial result with {len(result_text_so_far)} chars")
                return result_text_so_far
            # If no partial results, return empty to trigger fallback
            return ""
        raise
        
    # Return the result from _consume(), or fallback to joined text_parts
    return result_text if 'result_text' in locals() else "".join(text_parts)

