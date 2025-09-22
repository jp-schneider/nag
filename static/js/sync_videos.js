document.addEventListener('DOMContentLoaded', (event) => {
    // 1. Find all independent synchronization groups
    const syncGroups = document.querySelectorAll('.sync-group');
    
    // Object to track video timers and prevent race conditions
    const videoTimers = new Map(); 

    // Function to set up the sync logic for a single group
    const setupSyncGroup = (group) => {
        const groupVideos = group.querySelectorAll('.sync-video');
        
        if (groupVideos.length < 2) {
            return;
        }

        const masterVideo = group.querySelector('.master-video') || groupVideos[0];
        const slaveVideos = Array.from(groupVideos).filter(v => v !== masterVideo);

        // Helper function for time sync
        const syncTime = () => {
            slaveVideos.forEach(slave => {
                if (slave.readyState >= 2 && Math.abs(slave.currentTime - masterVideo.currentTime) > 0.1) {
                    slave.currentTime = masterVideo.currentTime;
                }
            });
        };
        
        // --- Standard Sync Listeners (Play, Pause, Seeking) ---
        
        masterVideo.addEventListener('play', () => {
            slaveVideos.forEach(slave => { slave.play(); });
        });
        masterVideo.addEventListener('pause', () => {
            slaveVideos.forEach(slave => { slave.pause(); });
        });
        masterVideo.addEventListener('timeupdate', syncTime);
        masterVideo.addEventListener('seeking', syncTime);

        // Slave Listeners
        slaveVideos.forEach(slave => {
            slave.addEventListener('play', (e) => {
                e.preventDefault(); 
                masterVideo.currentTime = slave.currentTime;
                masterVideo.play();
            });
            slave.addEventListener('pause', () => { masterVideo.pause(); });
            slave.addEventListener('seeking', () => { masterVideo.currentTime = slave.currentTime; });
        });
    };

    // --- Intersection Observer ---
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const group = entry.target;
            const masterVideo = group.querySelector('.master-video') || group.querySelector('.sync-video');
            if (!masterVideo) return;

            // 1. Clear any pending play commands for this video group
            if (videoTimers.has(group)) {
                clearTimeout(videoTimers.get(group));
                videoTimers.delete(group);
            }

            if (entry.isIntersecting) {
                // 2. Schedule play command with a slight delay (e.g., 200ms)
                // This prevents race conditions during quick scrolling jitter
                const playTimer = setTimeout(() => {
                    if (masterVideo.paused) { 
                        masterVideo.play().catch(e => {
                            // CATCH and SILENCE the harmless AbortError
                            if (e.name !== 'AbortError') {
                                console.error("Video Playback Error:", e);
                            }
                        });
                    }
                    videoTimers.delete(group); // Clean up map after successful play attempt
                }, 200); // Wait 200ms before playing

                videoTimers.set(group, playTimer);
                
            } else {
                // 3. Pause immediately when scrolling out of view
                masterVideo.pause();
            }
        });
    }, {
        root: null, 
        threshold: 0.3
    });

    // 4. Apply sync logic and observer to all groups
    syncGroups.forEach(group => {
        setupSyncGroup(group); 
        observer.observe(group);
    });
});