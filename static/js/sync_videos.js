
function is_group_ready(group) {
    return new Promise(resolve => {
        const groupVideos = group.querySelectorAll('.sync-video');
        let videosLoaded = 0;
        const totalVideos = groupVideos.length;

        if (totalVideos === 0) {
            resolve();
            return;
        }

        groupVideos.forEach(video => {
            if (video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA) { // readyState 4 means `HAVE_ENOUGH_DATA`
                videosLoaded++;
                //console.log("Video is loaded HAVE_ENOUGH_DATA: ", video.id)
            } else {
                video.addEventListener('canplaythrough', () => {
                    videosLoaded++;
                    //console.log("Video is loaded canplaythrough: ", video.id)
                    if (videosLoaded === totalVideos) {
                        resolve();
                    }
                }, { once: true }); // Use {once: true} to prevent re-firing
                video.load(); // Force the browser to start loading the video
                video.pause(); // Ensure the video is paused
            }
        });
    });
};

function startCarouselAutoplayWhenReady(carouselInstance) {
    // Find the currently active item within the carousel
    const activeItem = carouselInstance.items.find(item => item.classList.contains('is-active'));
    if (!activeItem) {
        return;
    }

    // Find all video sync groups within this active carousel item
    const allVideoGroupsInItem = activeItem.querySelectorAll('.sync-group');
    if (allVideoGroupsInItem.length === 0) {
        carouselInstance.start();
        return;
    }

    // Create a promise for each video group
    const groupPromises = Array.from(allVideoGroupsInItem).map(group => is_group_ready(group));

    // Wait for all promises to resolve
    Promise.all(groupPromises)
        .then(() => {
            // Once all groups are ready, start the carousel's autoplay
            console.log("All video groups ready, starting carousel autoplay of carousel: ", carouselInstance.id);
            carouselInstance.start();
        })
        .catch(error => {
            console.error("An error occurred while loading video groups:", error);
        });
};

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

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: false,
		autoplaySpeed: 10000,
    }

	// Initialize all div with carousel class
    var textureEditCarousel = bulmaCarousel.attach('#texture-edit-carousel', options)[0];
    var qualCarousel = bulmaCarousel.attach('#qualitative-carousel', options)[0];
    var qualCarouselDavis = bulmaCarousel.attach('#qualitative-carousel-davis', options)[0];
    var qualCarouselDecomp = bulmaCarousel.attach('#qualitative-carousel-decomp', options)[0];

    bulmaSlider.attach();


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
                is_group_ready(group).then(() => {
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
                });
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

    //startCarouselAutoplayWhenReady(textureEditCarousel)


});